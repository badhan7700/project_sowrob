# Traffic Sign Recognition Web App

Web-based traffic sign classification using GTSRB dataset models.

## Models
- **Simple CNN**: 32×32 input, fast inference
- **MobileNetV2**: 96×96 input, higher accuracy

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser.

## Usage
1. Upload a traffic sign image (JPG/PNG)
2. Select a model
3. Click "Recognize Sign"
4. View top-3 predictions with confidence scores
