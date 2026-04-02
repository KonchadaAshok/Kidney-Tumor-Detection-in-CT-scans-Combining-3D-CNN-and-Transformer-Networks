"""
Flask Web Application for Kidney Disease Classification
Provides a web interface for image upload and inference
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import warnings
import os
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import json
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r"model/hybrid_model_best.pth"
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
IMG_SIZE = 224

# Global model cache
model_cache = {'model': None, 'device': DEVICE}

# ==================== MODEL DEFINITIONS ====================

class CNNClassificationModel(nn.Module):
    """Lightweight CNN for image classification"""
    
    def __init__(self, num_classes=4, base_channels=32):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(base_channels*8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head attention"""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerClassifier(nn.Module):
    """Vision Transformer for image classification"""
    
    def __init__(self, img_size=224, patch_size=16, num_classes=4, embed_dim=256, depth=4, num_heads=8):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed_dim = 3 * patch_size * patch_size
        
        self.patch_embed = nn.Linear(self.patch_embed_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=4.0,
                dropout=0.1
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.reshape(batch_size, 3, self.img_size // self.patch_size, self.patch_size,
                      self.img_size // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(batch_size, self.num_patches, self.patch_embed_dim)
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


class HybridCNNTransformerModel(nn.Module):
    """Hybrid model combining CNN feature extraction with Transformer classification"""
    
    def __init__(self, num_classes=4, cnn_base_channels=32, transformer_embed_dim=256):
        super().__init__()
        self.num_classes = num_classes
        
        self.cnn_features = nn.Sequential(
            nn.Conv2d(3, cnn_base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_base_channels, cnn_base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(cnn_base_channels, cnn_base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_base_channels*2, cnn_base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(cnn_base_channels*2, cnn_base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_base_channels*4, cnn_base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        cnn_feature_channels = cnn_base_channels * 4
        
        self.cnn_to_embedding = nn.Sequential(
            nn.Conv2d(cnn_feature_channels, transformer_embed_dim, kernel_size=1),
            nn.BatchNorm2d(transformer_embed_dim)
        )
        
        self.num_transformer_blocks = 2
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=transformer_embed_dim,
                num_heads=8,
                mlp_ratio=4.0,
                dropout=0.1
            ) for _ in range(self.num_transformer_blocks)
        ])
        
        self.norm = nn.LayerNorm(transformer_embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(transformer_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        cnn_out = self.cnn_features(x)
        cnn_out = self.cnn_to_embedding(cnn_out)
        batch_size, embed_dim, height, width = cnn_out.shape
        cnn_out = cnn_out.view(batch_size, embed_dim, -1).permute(0, 2, 1)
        
        for block in self.transformer_blocks:
            cnn_out = block(cnn_out)
        
        cnn_out = self.norm(cnn_out)
        cnn_out = cnn_out.mean(dim=1)
        out = self.classifier(cnn_out)
        return out


# ==================== UTILITY FUNCTIONS ====================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path):
    """Load model from disk (cached)"""
    if model_cache['model'] is not None:
        return model_cache['model']
    
    device = DEVICE
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Auto-detect model type
    if 'hybrid' in model_path.lower():
        model = HybridCNNTransformerModel(num_classes=len(CLASS_NAMES))
    elif 'transformer' in model_path.lower():
        model = VisionTransformerClassifier(num_classes=len(CLASS_NAMES))
    else:
        model = CNNClassificationModel(num_classes=len(CLASS_NAMES))
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    model_cache['model'] = model
    return model


def preprocess_image(image_path, img_size=224):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image


def run_inference(image_path):
    """Run inference on image"""
    try:
        model = load_model(MODEL_PATH)
        image_tensor, pil_image = preprocess_image(image_path, IMG_SIZE)
        image_tensor = image_tensor.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_idx = predicted_class.item()
        predicted_name = CLASS_NAMES[predicted_idx]
        confidence_score = confidence.item()
        all_confidences = probabilities[0].cpu().numpy()
        
        results = {
            'predicted_class': predicted_name,
            'predicted_index': predicted_idx,
            'confidence': float(confidence_score),
            'confidences': {CLASS_NAMES[i]: float(all_confidences[i]) for i in range(len(CLASS_NAMES))},
            'success': True
        }
        
        return results
    
    except Exception as e:
        return {'success': False, 'error': str(e)}


def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    # Convert to RGB if image is in palette mode or has alpha channel
    if image.mode in ('P', 'RGBA', 'LA', 'PA'):
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    """Home page with upload form"""
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    
    return render_template('index.html', 
                         gpu_available=gpu_available, 
                         device_name=device_name,
                         classes=CLASS_NAMES)


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    
    # Check if image file is in request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file format'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run inference
        results = run_inference(filepath)
        
        if not results['success']:
            return jsonify(results), 400
        
        # Read image and convert to base64
        with Image.open(filepath) as img:
            img_base64 = image_to_base64(img)
        
        results['image'] = img_base64
        results['filename'] = filename
        
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'device': str(DEVICE).replace('cuda', '').strip('()') or 'cuda',
        'model_loaded': model_cache['model'] is not None
    }), 200


@app.route('/api/contact', methods=['POST'])
def contact():
    """Handle contact form submissions"""
    try:
        # Get form data
        name = request.form.get('name', 'Anonymous')
        email = request.form.get('email', '')
        message = request.form.get('message', '')
        
        # Validate
        if not email or not message:
            return jsonify({'success': False, 'error': 'Email and message required'}), 400
        
        # Log contact message (in production, send email or store in database)
        print("\n" + "="*60)
        print("NEW CONTACT MESSAGE")
        print("="*60)
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Message: {message}")
        print("="*60 + "\n")
        
        return jsonify({'success': True, 'message': 'Thank you for contacting us!'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large"""
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Kidney Disease Classification - Web Interface")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("="*60)
    print("\nStarting Flask server...")
    print("Open your browser to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
