# Wake Word Detector + On Device ML

This project involved manually collecting data for wake word detection, preprocessing said data, and training a custom built CNN model architecture for binary classification.

### utils.py 
contains the code for (1) collecting the audio data from the computer microphone (200 positive samples, 1000 negative samples), (2) data augmentation techniques including frequency and time shifting/masking, aas well as pitch shift. 
This file contains the method that creates the DataLoader for our model training.

### cnn.py
this file contains the implementation and training of the CNN model. Hyperparameter search was conducted by running model training with a grid search to find optimal parameters for learning rate, weight decay, label weights, etc.

### model_evaluation.ipynb
this file contains code that evaluated the model on a test set. Ideally, since the dataset is small, one would perform K fold cross validation to get a measure of the model performance, with enough K folds to also create a confidence interval around relevant metrics. For the sake of brevity, this was not done. 

### quantization.py
this file prepares the model for efficient mobile inference


<img width="250" alt="Screen Shot 2023-04-03 at 9 23 58 AM" src="https://user-images.githubusercontent.com/22806151/230750444-d8b420ab-e93a-4358-aba5-abd5efc90e3c.png">

model training

<img width="507" alt="Screen Shot 2023-04-03 at 10 39 46 AM" src="https://user-images.githubusercontent.com/22806151/230750448-75b129f6-4635-4ceb-8315-54ffadd2a807.png">

model evaluation


# WAKEWORD APP
<img width="351" alt="Screen Shot 2023-04-08 at 7 11 44 PM" src="https://user-images.githubusercontent.com/22806151/230750641-e0e6381d-6498-4b5e-97d8-1e71be24e93e.png">

The model was deployed on an Android device. File wakeapp/app/src/main/java/example/MainActivity.java contains the main code for the application. 




