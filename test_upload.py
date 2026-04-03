import requests
import sys
import os
from pathlib import Path
import time

# Wait for Flask to be ready
time.sleep(2)

# Find a test image
test_dir = Path("Dataset/Normal")
if not test_dir.exists():
    print("ERROR: Dataset/Normal directory not found")
    sys.exit(1)

# Get the first image
image_files = list(test_dir.glob("*.jpg"))
if not image_files:
    print(f"ERROR: No JPG images found in {test_dir}")
    sys.exit(1)

test_image = image_files[0]
print(f"OK: Found test image: {test_image.name}")
print(f"    File size: {test_image.stat().st_size / 1024:.2f} KB")

# Try uploading to the API
url = "http://localhost:5000/api/predict"
try:
    with open(test_image, 'rb') as f:
        files = {'file': f}
        print(f"\nUploading to {url}...")
        response = requests.post(url, files=files, timeout=60)
    
    print(f"API Response Status: {response.status_code}")
    print(f"Response Content-Type: {response.headers.get('content-type')}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nSUCCESS: Upload and prediction successful!")
        print(f"   Predicted class: {data.get('predicted_class')}")
        print(f"   Confidence: {data.get('confidence'):.2%}")
        print(f"   All classes:")
        for cls, conf in data.get('all_confidences', {}).items():
            print(f"     - {cls}: {conf:.2%}")
    else:
        print(f"ERROR Status {response.status_code}: {response.text}")
    
except requests.exceptions.ConnectionError:
    print(f"ERROR: Could not connect to Flask server at {url}")
    print(f"   Make sure the Flask app is running: python app.py")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
