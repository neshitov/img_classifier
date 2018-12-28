# Categorical image classifier implementation in PyTorch
## Installations.
Project consists of python scripts for image classification.
- standard libraries numpy, pytorch-0.4, PIL and matplotlib

## Files description.
There are two python scripts: train.py does the training of image classificator and predict.py does inference using the model created by train.py. In train.py we take a pretrained convolutional network layer in torchvision, freeze its feature part and replace the classifier part with a new classifier that consists of one fully connected hidden layer and the output layer with dimenion equal to the number of image classes. The classifier part is then trained on the dataset. Number of image classes is determined automativally as the number of subfloders in the training dataset. Each output node represents the log-probability that the input image belongs to the corresponding class.

The script train.py takes the following parameters:
- data_dir: path to the data directory. The data directory is expected to have subfolders containing images of each type.
Optional:
-- save_dir: path where to save the trained model. If None, current directory will be used
-- learning_rate: learning rate. If None, lr=0.001
-- hidden_units: number of hidden units in the classification layer of the network. If None, 1024 hidden units are used
-- epochs: number of epochs to train the model. If None, model will be trained for 3 epochs
-- arch: pretrained network architecture to be used. If None, VGG19 with batch normalization will be used
-- gpu: id --gpu is added, the model will be trained using gpu

## Results
Classification model is traind, validation set accuracy is computed. The script predict.py is used to make inference with trained model.
