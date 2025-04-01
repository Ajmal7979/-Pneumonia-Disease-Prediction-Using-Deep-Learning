# -Pneumonia-Disease-Prediction-Using-Deep-Learning
Developed a deep learning-based model for pneumonia detection using chest X-ray images.


# Pneumonia Disease Prediction using Deep Learning

## Overview
This project aims to develop a deep learning-based system for detecting pneumonia from chest X-ray images. The model utilizes state-of-the-art convolutional neural network (CNN) architectures such as **VGG16, ResNet50, and DenseNet121** to classify images as normal or pneumonia-affected.

## Technologies Used
- **Programming Language:** Python
- **Libraries & Frameworks:** TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn
- **Deep Learning Models:** VGG16, ResNet50, DenseNet121
- **Dataset:** Chest X-ray images from the **Kaggle Pneumonia Dataset**

## Features
- **Preprocessing:** Image resizing, normalization, and augmentation to improve model performance.
- **Model Training:** Implementation of VGG16, ResNet50, and DenseNet121 for pneumonia classification.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- **Deployment:** Flask-based API for real-time prediction.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow & Keras
- OpenCV
- NumPy, Pandas, Matplotlib, Scikit-learn

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/pneumonia-prediction.git
   cd pneumonia-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download and place the dataset in the `data/` folder.

## Model Training
Run the following command to train the models:
```sh
python train.py --model vgg16  # or resnet50, densenet121
```

## Testing the Model
To evaluate the trained model:
```sh
python test.py --model vgg16
```

## Deployment
To start the Flask-based API for real-time predictions:
```sh
python app.py
```
Access the web interface at `http://127.0.0.1:5000/`

## Results
| Model      | Accuracy | Precision | Recall | F1-Score |
|------------|---------|-----------|--------|----------|
| VGG16      | 92.5%   | 91.3%     | 92.8%  | 92.0%    |
| ResNet50   | 94.2%   | 93.5%     | 94.7%  | 94.1%    |
| DenseNet121| 95.1%   | 94.8%     | 95.5%  | 95.1%    |

## Future Improvements
- Implementing attention mechanisms for better feature extraction.
- Extending the model for multi-class classification of lung diseases.
- Deploying the model as a cloud-based service.


