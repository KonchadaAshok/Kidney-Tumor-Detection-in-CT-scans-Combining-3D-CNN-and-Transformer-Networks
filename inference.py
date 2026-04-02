import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Define paths as variables - MODIFY THESE
# Use forward slashes or raw strings for Windows paths
IMAGE_PATH = r"sample images\Cyst\image.jpg"  # Change to your image path
MODEL_PATH = r"model\hybrid_model_best.pth"     # Change to your model path
DEVICE = torch.device('cpu')                    # Use 'cuda' if you have GPU

# Class mapping
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
IMG_SIZE = 224

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
        
        # CNN Feature Extractor (lighter version)
        self.cnn_features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, cnn_base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_base_channels, cnn_base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            
            # Block 2
            nn.Conv2d(cnn_base_channels, cnn_base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_base_channels*2, cnn_base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            
            # Block 3
            nn.Conv2d(cnn_base_channels*2, cnn_base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_base_channels*4, cnn_base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
        )
        
        # CNN features will be: (batch, 128, 28, 28)
        cnn_feature_channels = cnn_base_channels * 4
        
        # Project CNN features to embedding dimension
        self.cnn_to_embedding = nn.Sequential(
            nn.Conv2d(cnn_feature_channels, transformer_embed_dim, kernel_size=1),
            nn.BatchNorm2d(transformer_embed_dim)
        )
        # After this: (batch, transformer_embed_dim, 28, 28)
        # Flatten to patches: (batch, 784, transformer_embed_dim) for 28x28 patch grid
        
        # Transformer blocks for fusion
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
        
        # Fusion and classification head
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
        # CNN feature extraction
        cnn_out = self.cnn_features(x)  # (batch, 128, 28, 28)
        
        # Project to embedding dimension
        cnn_out = self.cnn_to_embedding(cnn_out)  # (batch, embed_dim, 28, 28)
        
        # Reshape for transformer: treat spatial features as sequence
        batch_size, embed_dim, height, width = cnn_out.shape
        # Flatten spatial dimensions to sequence
        cnn_out = cnn_out.view(batch_size, embed_dim, -1).permute(0, 2, 1)  # (batch, 784, embed_dim)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            cnn_out = block(cnn_out)  # (batch, 784, embed_dim)
        
        # Apply layer norm
        cnn_out = self.norm(cnn_out)
        
        # Global average pooling over sequence dimension
        cnn_out = cnn_out.mean(dim=1)  # (batch, embed_dim)
        
        # Classification
        out = self.classifier(cnn_out)
        return out


# ==================== PREPROCESSING ====================

def load_and_preprocess_image(image_path, img_size=224):
    """Load and preprocess image for inference"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms (same as used during training)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


# ==================== INFERENCE FUNCTION ====================

def run_inference(image_path, model_path, device=DEVICE, img_size=IMG_SIZE):
    """
    Run inference on an image with a trained model
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the saved model
        device (torch.device): Device to run inference on
        img_size (int): Input image size
    
    Returns:
        dict: Prediction results with class name, probability, and all class confidences
    """
    
    try:
        # Check if image exists
        if not Path(image_path).exists():
            print(f"❌ Error: Image not found at {image_path}")
            return None
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ Error: Model not found at {model_path}")
            return None
        
        # Load and preprocess image
        print(f"📷 Loading image from: {image_path}")
        image_tensor = load_and_preprocess_image(image_path, img_size)
        image_tensor = image_tensor.to(device)
        
        # Determine model type from path
        if 'hybrid' in model_path.lower():
            model = HybridCNNTransformerModel(num_classes=len(CLASS_NAMES))
        elif 'transformer' in model_path.lower():
            model = VisionTransformerClassifier(img_size=img_size, num_classes=len(CLASS_NAMES))
        else:  # CNN by default
            model = CNNClassificationModel(num_classes=len(CLASS_NAMES))
        
        # Load model weights
        print(f"🤖 Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Run inference
        print("⚙️  Running inference...")
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Prepare results
        predicted_idx = predicted_class.item()
        predicted_name = CLASS_NAMES[predicted_idx]
        confidence_score = confidence.item()
        
        # Get all class confidences
        all_confidences = probabilities[0].cpu().numpy()
        
        results = {
            'predicted_class': predicted_name,
            'predicted_index': predicted_idx,
            'confidence': confidence_score,
            'all_confidences': {CLASS_NAMES[i]: float(all_confidences[i]) for i in range(len(CLASS_NAMES))}
        }
        
        return results
    
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        return None


# ==================== DISPLAY RESULTS ====================

def display_results(results):
    """Display inference results in a formatted way"""
    
    if results is None:
        return
    
    print("\n" + "="*60)
    print("📊 INFERENCE RESULTS")
    print("="*60)
    print(f"\n✅ Predicted Class: {results['predicted_class']}")
    print(f"📈 Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
    print(f"\nAll Class Confidences:")
    for class_name, conf in results['all_confidences'].items():
        bar_length = int(conf * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {class_name:10s} [{bar}] {conf:.4f} ({conf*100:.2f}%)")
    print("="*60 + "\n")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n🏥 Kidney Disease Classification - Inference")
    print("="*60)
    
    # Run inference
    results = run_inference(IMAGE_PATH, MODEL_PATH, device=DEVICE, img_size=IMG_SIZE)
    
    # Display results
    if results:
        display_results(results)
        print(f"✨ Inference completed successfully!")
    else:
        print("❌ Inference failed!")
