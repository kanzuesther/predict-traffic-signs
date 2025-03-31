# Traffic Sign Recognition System

This project implements a Convolutional Neural Network (CNN) for traffic sign recognition using TensorFlow. It includes both a training script and a graphical user interface for making predictions.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV (cv2)
- NumPy
- scikit-learn
- Pillow (PIL)
- tkinter (usually comes with Python)

Install dependencies:
```bash
pip install tensorflow opencv-python numpy scikit-learn pillow
```

## Project Structure

- `traffic.py`: Training script for the CNN model
- `predict_sign.py`: GUI application for making predictions using trained models
- Dataset should be organized in numbered folders (0-42) corresponding to different traffic sign categories

## Usage

### Training the Model

1. Organize your dataset with numbered folders (0-42) containing traffic sign images
2. Run the training script:
```bash
python traffic.py /path/to/dataset [output_model.h5]
```

### Making Predictions

1. Run the GUI application:
```bash
python predict_sign.py
```
2. Click "Browse" to load a trained model (.h5 file)
3. Click "Select Image" to choose a traffic sign image
4. View the prediction results

## Model Architecture

The CNN model consists of:
- 3 convolutional blocks with batch normalization and max pooling
- Flatten layer
- 2 dense layers with dropout for regularization
- Output layer with 43 categories

## Training Parameters

- Image dimensions: 30x30 pixels
- Number of categories: 43
- Training/test split: 60%/40%
- Validation split: 20% of training data
- Early stopping with patience of 3 epochs
- Maximum epochs: 10
