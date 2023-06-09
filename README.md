# Part 1 - Back Propogation



# Part 2 - MNIST classification

## Description

This project includes two Python files: `model.py`, and `S5.ipynb`. These files are part of a machine learning project for image classification using the MNIST dataset. The project is about training a neural network model to recognize handwritten digits.

Few samples in the dataset are shown below. [Image taken from S5 assignment]

![MNIST](Test_Images/train_data_sample.png)


## Files

### 1. model.py

This file defines the structure of the neural network model used for image classification. The `Net` class is a subclass of `torch.nn.Module` and consists of several convolutional and fully connected layers. The `forward` method implements the forward pass of the model, and the `summary` method provides a summary of the model's architecture. The network architecture uses batch normalization as part of regularization and uses dropout of 0.30 so that the model gives an improved test accuracy. As we find from the model summary, the total number of parameters this model uses is around 17.5K. 

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
            Conv2d-4           [-1, 32, 28, 28]           4,640
              ReLU-5           [-1, 32, 28, 28]               0
       BatchNorm2d-6           [-1, 32, 28, 28]              64
         MaxPool2d-7           [-1, 32, 14, 14]               0
            Conv2d-8            [-1, 8, 14, 14]             264
           Dropout-9            [-1, 8, 14, 14]               0
           Conv2d-10           [-1, 16, 12, 12]           1,168
             ReLU-11           [-1, 16, 12, 12]               0
      BatchNorm2d-12           [-1, 16, 12, 12]              32
           Conv2d-13           [-1, 32, 10, 10]           4,640
             ReLU-14           [-1, 32, 10, 10]               0
      BatchNorm2d-15           [-1, 32, 10, 10]              64
           Conv2d-16            [-1, 8, 10, 10]             264
          Dropout-17            [-1, 8, 10, 10]               0
           Conv2d-18             [-1, 16, 8, 8]           1,168
             ReLU-19             [-1, 16, 8, 8]               0
           Conv2d-20             [-1, 32, 6, 6]           4,640
             ReLU-21             [-1, 32, 6, 6]               0
           Conv2d-22             [-1, 10, 6, 6]             330
        AvgPool2d-23             [-1, 10, 1, 1]               0
================================================================
Total params: 17,466
Trainable params: 17,466
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.11
Params size (MB): 0.07
Estimated Total Size (MB): 1.18
----------------------------------------------------------------
```

### 3. S6.ipynb

The `S6.ipynb` file is the main module that runs the image classification activity for MNIST dataset. It contains the code for training and evaluating a neural network model using the MNIST dataset. The file includes the following components:

        -       Importing necessary libraries and dependencies
        -       Mounting Google Drive
        -       Setting up the device (CPU or GPU)
        -       Defining data transformations for training and testing
        -       Loading the MNIST dataset
        -       Setting up data loaders
        -       Instantiate the neural network model and displaying its summary
        -       Training the model using SGD optimizer and NLL loss
        -       Displaying test logs

This file trains the model with a learning rate of 0.1 using batch size of 256. After training this model for less than 20 epochs, the trained model touches test accuracy of 99.44% at the 13th epoch, gives fluctucating accuracy till 19th epoch and at the 19th epoch gives the test accuracy of 99.48%.
Please note that this README is dynamically generated and serves as a placeholder. As you make further modifications to the project, remember to update this file accordingly. Provide a brief description of each file, its purpose, and its relevance to the project's functionality.

For more detailed information on the project's implementation and code, please refer to the individual files mentioned above.

## Usage

To run the project, make sure you have the dependencies installed.
```
pip install -r requirements.txt
```
You can execute the `S5.ipynb` notebook to perform the training and testing. Adjust the hyperparameters such as learning rate, momentum, batch size, and number of epochs to improve the model performance as desired.

Below is the sample output that can be found in S6.ipnyb.

### 1. Sample Test Stats

![Test Log](Test_Images/Test_inference_result_b256.png)

