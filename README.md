# SMART_GALLERY-TASK

# FER2013 Emotion Recognition Model

This repository implements an emotion recognition model using the FER2013 dataset. The model is built using a custom convolutional neural network (CNN) and is trained to classify facial expressions into one of seven emotions.

1. CUSTOM MODEL: https://www.kaggle.com/code/sarthshah777/cnn-task
2. Transfer Learning Model : https://www.kaggle.com/code/sarthshah777/cnn-task-transfer-learning
   

## Approach and Decisions

### 1. **Preprocessing and Augmentation**
- **Grayscale to RGB:** Since FER2013 images are grayscale, they were converted to RGB to match the input requirements of the custom CNN model.
- **Data Augmentation:** Applied random transformations to the images, such as horizontal flipping, random rotation, and color jitter (brightness, contrast, saturation, and hue), to help the model generalize better.

### 2. **Model Selection**
The model used is a custom CNN (`MyModel`), which consists of:
- 4 convolutional layers with increasing channels (32, 64, 128, 256).
- Dropout layers to prevent overfitting.
- Fully connected layers to classify the images into 7 emotion classes from the FER2013 dataset.

### 3. **Loss Function and Optimizer**
- **Loss Function:** `CrossEntropyLoss` for multi-class classification.
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum to update model weights.
- **Learning Rate Scheduler:** `StepLR` to reduce the learning rate by a factor of 10 every 7 epochs.

### 4. **Training**
- The model was trained for 25 epochs, alternating between training and validation phases.
- The model with the highest validation accuracy was saved.

5. **Model Evaluation:**
    - After training, the model will save the best weights based on validation accuracy. You can evaluate the model's performance on the test set or use it for inference.


   

2.  Transfer Learning Method 

code:
https://www.kaggle.com/code/sarthshah777/cnn-task-transfer-learning/edit

Approach and Decisions:

Preprocessing and Augmentation:

Converted grayscale images to RGB since ResNet-18 expects RGB image.

Data Augmentation: Random horizontal flip, rotation, color jitter to enhance the generalization.

Model Selection:
ResNet-18 with pre-trained weights, last layer changed accordingly to output 7 classes since FER2013 has only 7 emotions.

Loss Function and Optimizer:
CrossEntropyLoss for multi-class classification.

Optimizer: SGD with momentum
 Scheduler: StepLR γ = 0.1 periodicity = 7 epochs
 
 Training:
 
 Trained for 25 epochs with a training and then validation phase. Saved the model with best validation accuracy
 

  Challenges

 Data imbalance - FER2013 has class imbalance which can induce bias
 
 Computational constraints - Training on limited resources may take considerable time
 
 Overfitting: Augmentations helped in this case, but dropout might be helpful for this model as well.
