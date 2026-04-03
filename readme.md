# KidneyAI Diagnostics - AI-Powered Kidney Disease Classification

A professional web application that uses a hybrid CNN + Transformer deep learning model to classify kidney ultrasound images into four categories: **Cyst, Normal, Stone, and Tumor**.

## 🎯 Features

- **Advanced AI Model**: Hybrid CNN + Transformer architecture for accurate kidney disease detection
- **Professional Web Interface**: Modern, responsive dashboard with real-time predictions
- **Fast Inference**: GPU-accelerated processing (with CPU fallback)
- **Contact Form**: Built-in communication system
- **Mobile-Friendly**: Fully responsive design with dynamic navbar
- **Real-Time Device Status**: Shows whether GPU or CPU is being used

## 📋 System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- GPU support (NVIDIA CUDA) optional but recommended
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## 📁 Project Structure

```
output/
├── app.py                    # Flask web server & ML inference pipeline
├── inference.py              # Model loading & prediction logic
├── requirements.txt          # Python dependencies
├── hybrid_model_best.pth    # Pre-trained model weights
├── templates/
│   └── index.html           # Main web interface
├── static/
│   ├── css/
│   │   └── style.css        # Professional styling
│   └── js/
│       └── script.js        # Interactive features
└── sample_images/           # Test images for demonstration
    ├── Cyst/
    ├── Normal/
    ├── Stone/
    └── Tumor/
```

## 💻 Usage

1. **Open** the application in your browser at `http://localhost:5000`
2. **Upload** a kidney ultrasound image (PNG, JPG, JPEG)
3. **Wait** for the AI to analyze the image
4. **View** the diagnosis result and confidence scores

## � Technical Details

### Model Architecture
- **Type**: Hybrid CNN + Transformer
- **Classes**: 4 (Cyst, Normal, Stone, Tumor)
- **Framework**: PyTorch 2.0+
- **Input**: Medical ultrasound images

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main web interface |
| `/api/predict` | POST | Image classification |
| `/api/health` | GET | Server health & device info |
| `/api/contact` | POST | Contact form submission |

### Supported Classes

| Class  | Description           |
|--------|----------------------|
| Cyst   | Kidney cyst detected  |
| Normal | No abnormality found  |
| Stone  | Kidney stone detected |
| Tumor  | Tumor detected        |

### Device Support
- **GPU (CUDA)**: Automatic detection and activation
- **CPU**: Fallback for systems without NVIDIA GPU
- Device status displayed in real-time

## 📧 Contact

Use the Contact form on the application dashboard to send inquiries.

---

**Status**: Production Ready ✅
