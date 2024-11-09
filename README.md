# SMART_GALLERY-TASK

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
