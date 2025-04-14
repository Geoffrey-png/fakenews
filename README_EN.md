# FakeNews Detection System

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Vue](https://img.shields.io/badge/vue-2.7+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.7+-red.svg)

A deep learning-based multi-modal fake news detection system that integrates text analysis and image tampering detection, providing users with comprehensive news authenticity analysis services.

English | [中文](README.md)

## 📋 Project Overview

This project is a complete fake news detection solution that uses the BERT model for text analysis, the HAMMER model for image tampering detection, and provides services to users through a friendly web interface. The system supports various functions such as single-text detection, batch text detection, image detection, and mixed detection.

The project aims to help users identify and analyze potentially false news content through deep learning and artificial intelligence technology, improve information filtering capabilities, and reduce the impact of false information spread.

### Core Features

- **Single Text Detection**: Analyze the authenticity of a single news text
- **Batch Text Detection**: Process multiple news texts simultaneously
- **Image Detection**: Identify various tampering traces in images
- **Mixed Detection**: Comprehensive analysis combining text and image
- **Fake News Explanation**: Generate explainable analysis results using large model APIs

## 🏗️ System Architecture

### Frontend Architecture
- **Framework**: Vue.js 2.x + Element UI
- **HTTP Client**: Axios
- **Router**: Vue Router

### Backend Architecture
- **Web Framework**: Flask + Flask-CORS
- **Deep Learning**: PyTorch + Transformers
- **Image Processing**: Pillow
- **NLP Processing**: jieba word segmentation
- **Deployment**: Gunicorn

### File Structure
```
fake_news_detection/
├── frontend/             # Frontend Vue application
├── api/                  # Backend API service
├── src/                  # Model training and evaluation code
├── hammer_detector/      # Image detection module
├── models/               # Pre-trained models
├── data/                 # Datasets
└── results/              # Results and visualizations
```

## 🚀 Installation and Deployment

### Environment Requirements
- Python 3.7+
- Node.js 12+
- CUDA 10.1+ (recommended for GPU acceleration)

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/Geoffrey-png/fakenews.git
cd fakenews
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install image detection module dependencies**
```bash
cd hammer_detector
pip install -r requirements.txt
cd ..
```

4. **Download pre-trained models**
```bash
# Download BERT model
mkdir -p models/bert_local
# Please download the bert-base-chinese model from the corresponding source

# Download HAMMER model weights
mkdir -p hammer_detector/weights
# Please download checkpoint_best.pth from the project release page
```

5. **Start the backend service**
```bash
cd api
bash start_server.sh
# or on Windows
python app.py
```

### Frontend Setup

1. **Install dependencies**
```bash
cd frontend
npm install
```

2. **Start the development server**
```bash
npm run serve
```

3. **Build for production**
```bash
npm run build
```

## 📝 User Guide

### Text Detection

1. Visit the system homepage
2. Select "Single Text Detection" or "Batch Detection" function
3. Enter or paste the news text to be detected
4. Click the "Start Detection" button
5. View detection results and analysis explanation

### Image Detection

1. Visit the system homepage
2. Select the "Image Detection" function
3. Upload the image to be analyzed
4. The system automatically performs tampering detection
5. View detection results and tampering area visualization

## 🔬 Technical Details

### Text Detection Model

- BERT-based binary classification model
- Pre-trained and fine-tuned using Chinese corpus
- Accuracy exceeding 96%
- Supports cache optimization and batch processing

### Image Detection Technology

- Multi-modal tampering detection based on the HAMMER framework
- Supports the recognition of facial, textual, and object tampering
- Provides high-precision tampering area visualization
- Supports classification of multiple tampering types

### API Interfaces

| Interface | Method | Description |
|-----|------|-----|
| `/health` | GET | Health check interface |
| `/predict` | POST | Single text detection interface |
| `/batch_predict` | POST | Batch text detection interface |
| `/detect/image` | POST | Image detection interface |
| `/generate_explanation` | POST | Fake news explanation generation interface |

## 📊 Examples

### Text Detection Results

![Text Detection Example](Figure_1.png)

### Image Detection Results

![Image Detection Example](Figure_2.png)

## 👨‍💻 Developer Guide

### Adding New Features

1. **Frontend Development**
   - Create new page components in the `frontend/src/views/` directory
   - Update `frontend/src/router/index.js` to add routes
   - Add new API calls in `frontend/src/utils/api.js`

2. **Backend Development**
   - Add new API endpoints in `api/app.py`
   - Extend functionality in existing modules or create new modules

### Model Training

```bash
bash run_train.sh
```
