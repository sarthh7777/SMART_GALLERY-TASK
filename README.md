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


## CHALLENGES 

1.Gradient Vanishing/Exploding:



2. Hyperparameter Tuning:

The number of layers, kernel size, dropout rate, and learning rate are critical hyperparameters. Finding the right combination for  custom model 
   

# Transfer Learning

## Approach and Decisions

### 1. **Preprocessing and Augmentation**
- **Grayscale to RGB:** FER2013 images are originally grayscale, so they were converted to RGB for compatibility with ResNet-18.
- **Data Augmentation:** Applied random transformations like horizontal flip, rotation, and color jitter (brightness, contrast, saturation, and hue) to improve the model's ability to generalize.

### 2. **Model Selection**
- **ResNet-18:** Used ResNet-18 with pre-trained weights on ImageNet for transfer learning. The final layer was modified to output 7 classes to match the 7 emotion classes in the FER2013 dataset.

### 3. **Loss Function and Optimizer**
- **Loss Function:** `CrossEntropyLoss` for multi-class classification.
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum to optimize the model.
- **Learning Rate Scheduler:** `StepLR` with a decay factor of 0.1 every 7 epochs to reduce the learning rate gradually.

### 4. **Training**
- The model was trained for 25 epochs, alternating between training and validation phases.
- The model with the highest validation accuracy was saved for future evaluation or inference.

## Challenges

- **Data Imbalance:** The FER2013 dataset contains class imbalance, where some emotion categories have more samples than others. This can lead to biases in the model’s predictions, requiring careful evaluation and potential adjustments (e.g., using weighted loss functions or data balancing).


