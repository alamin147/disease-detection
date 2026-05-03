# Plant Disease Detection System

A deep learning-based web application for detecting plant diseases, specifically focusing on tomato plant diseases. This project uses a Convolutional Neural Network (CNN) trained on a large dataset of plant images to classify healthy plants and identify various disease conditions.

##  Overview

The Plant Disease Detection System leverages deep learning and computer vision to provide accurate, real-time disease identification for agricultural applications. The system achieves **98-99% accuracy** across 10 tomato plant disease classes using a custom CNN architecture.

##  Model Performance

- **Training Accuracy**: 98.6%
- **Validation Accuracy**: 98.6%
- **Dataset Size**: 36,690 images (18,345 training, 18,345 validation)
- **Number of Classes**: 10 tomato disease categories
- **Model Parameters**: ~7.8M trainable parameters

### Classification Report Highlights
- **Precision**: 98-99% across all classes
- **Recall**: 97-100% across all classes
- **F1-Score**: 98-99% average

##  Supported Disease Classes

The model can classify the following conditions:

1. **Tomato___Bacterial_spot** - Precision: 98%, Recall: 99%
2. **Tomato___Early_blight** - Precision: 98%, Recall: 99%
3. **Tomato___Late_blight** - Precision: 98%, Recall: 98%
4. **Tomato___Leaf_Mold** - Precision: 99%, Recall: 100%
5. **Tomato___Septoria_leaf_spot** - Precision: 99%, Recall: 97%
6. **Tomato___Spider_mites Two-spotted_spider_mite** - Precision: 98%, Recall: 99%
7. **Tomato___Target_Spot** - Precision: 98%, Recall: 98%
8. **Tomato___Tomato_Yellow_Leaf_Curl_Virus** - Precision: 99%, Recall: 99%
9. **Tomato___Tomato_mosaic_virus** - Precision: 99%, Recall: 100%
10. **Tomato___healthy** - Precision: 100%, Recall: 98%

##  Model Architecture

The CNN model consists of:

- **5 Convolutional Blocks**:
  - Block 1: 32 filters, Conv2D + MaxPool2D
  - Block 2: 64 filters, Conv2D + MaxPool2D
  - Block 3: 128 filters, Conv2D + MaxPool2D
  - Block 4: 256 filters, Conv2D + MaxPool2D
  - Block 5: 512 filters, Conv2D + MaxPool2D

- **Dropout Layers**: 25% and 40% dropout for regularization

- **Fully Connected Layers**:
  - Dense layer: 1,500 units with ReLU activation
  - Output layer: 10 units with Softmax activation

- **Total Parameters**: 7,800,734

