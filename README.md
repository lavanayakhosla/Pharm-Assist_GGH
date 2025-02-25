# Pharm-Assist : The Pharmacists' Assistant 

# Introduction

Doctors in India are overworked and tend to write fast and unclear handwriting. This causes a problem for pharmacists as well as the general public to understand their prescriptions. This may lead to confusion and in worse cases , might lead to administration of the wrong medication. My solution “ Pharm-Assist” is a web application that converts doctors’ handwritten prescriptions to text. It uses a deep learning model combining Convolutional Neural Network (CNN) and Bidirectional Long Short Term Memory (BiLSTM) trained on EMNIST dataset to identify words from images. CNN has been created using TensorFlow and Keras. The web application will have an option for users to upload their prescription , which will recognise the text from the image and match it from the medicine database to create an order. 

# Design Idea and Approach
# Technology used- 
i. TensorFlow and Keras which are deep learning frameworks.
ii. EMNIST dataset for handwritten text in images.
iii. Google Colab for running python code. 

# Approach
A hybrid CNN+BiLSTM model combines CNN (Convolutional Neural Network) for extracting features, while BiLSTM (Bidirectional Long Short-Term Memory) layers help in sequence modeling.
A clear technical breakdown of the designed model is as follows-

1. CNN Layers (Feature Extraction)
Each CNN layer extracts spatial features like edges, curves and textures which help in recognizing handwritten characters. CNN reduces the input complexity before inputting it to the LSTM.

2. The layers used:
Conv2D(32, (3,3), activation='relu')- Extracts features like strokes, edges.
BatchNormalization()- Normalizes activations for stable training.
MaxPooling2D(2,2)- Reduces dimensions, retaining only important features.
Dropout(0.25)- Prevents overfitting by randomly deactivating neurons.
BiLSTM Layers (Sequence Learning)
LSTM helps in modelling sequential interdependencies between the strokes in handwritten text.                 Two Bidirectional LSTM (BiLSTM) layers with 128 units each are used-
Bidirectional(LSTM(128, return_sequences=True))- Captures relationships between strokes of a character.
Bidirectional(LSTM(128))- Further refines understanding.
Fully Connected Layers
Dense layer with 256 neurons (ReLU activation)
Dropout for reducing overfitting
Output layer with 47 neurons (Softmax activation for classification)

3. Dominant Scaling Parameters-
Optimizer: Adam (adaptive learning rate)
Loss Function: Categorical Crossentropy
Batch Size: 128
Epochs: 30
Validation Set: 20% of training data used for validation







