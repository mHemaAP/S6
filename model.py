### Import PyTorch library components, various layers/functions from torch 
### Import torch NN functional modules
### Import the torchsummary library for model summary generation 
### Import the tqdm library for progress bar visualization during training

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm

# Define a neural network model called Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define the first convolutional block (conv1)
        self.conv1 = nn.Sequential(
          # 2D convolution with 1 input channel, 16 output channels, 
          # and 3x3 kernel size  
          nn.Conv2d(1, 16, 3, padding=1), # n_in = 28, n_out = 28
          nn.ReLU(), # ReLU activation function
          nn.BatchNorm2d(16),  # Batch normalization
          # 2D convolution with 16 input channels, 32 output channels, 
          # and 3x3 kernel size
          nn.Conv2d(16, 32, 3, padding=1), # n_in = 28, n_out = 28
          nn.ReLU(), # ReLU activation function
          nn.BatchNorm2d(32), # Batch normalization
          # Max pooling with 2x2 kernel size and stride 2
          nn.MaxPool2d(2, 2), # n_in = 28, n_out = 14
          # 2D convolution with 32 input channels, 8 output channels, 
          # and 1x1 kernel size. This step is to reduce the number of 
          # channels after combining all the features extracted till this point
          nn.Conv2d(32, 8, 1),
          # Apply regularization to improve accuracy
          # Dropout layer with dropout probability of 0.30
          nn.Dropout(0.30)   
        )

        # Define the second convolutional block (conv2)
        self.conv2 = nn.Sequential(
          # 2D convolution with 8 input channels, 16 output channels, 
          # and 3x3 kernel size  
          nn.Conv2d(8, 16, 3), # n_in = 14, n_out = 12
          nn.ReLU(),   # ReLU activation function
          nn.BatchNorm2d(16),   # Batch normalization
          # 2D convolution with 16 input channels, 32 output channels, 
          # and 3x3 kernel size
          nn.Conv2d(16, 32, 3), # n_in = 12, n_out = 10
          nn.ReLU(), # ReLU activation function
          nn.BatchNorm2d(32), # Batch normalization
          #nn.MaxPool2d(2, 2) # 10
          # 2D convolution with 32 input channels, 8 output channels, 
          # and 1x1 kernel size
          nn.Conv2d(32, 8, 1),  
          # Dropout layer with dropout probability of 0.30
          nn.Dropout(0.30)   
        )

        # Define the third convolutional block (conv3)
        self.conv3 = nn.Sequential(
          # 2D convolution with 8 input channels, 16 output channels, 
          # and 3x3 kernel size  
          nn.Conv2d(8, 16, 3), # n_in = 10, n_out = 8
          nn.ReLU(), # ReLU activation function
          # 2D convolution with 16 input channels, 32 output channels, 
          # and 3x3 kernel size
          nn.Conv2d(16, 32, 3), # n_in = 8, n_out = 6
          nn.ReLU(), # ReLU activation function
          # 2D convolution with 32 input channels, 10 output channels, 
          # and 1x1 kernel size
          nn.Conv2d(32, 10, 1), 
          # Average pooling with 6x6 kernel size
          nn.AvgPool2d(6)
        )        

    # Define the forward pass of the model
    def forward(self, x):
        # Apply conv1 to the input
        x = self.conv1(x)
        # Apply conv2 to the output of conv1
        x = self.conv2(x)
        # Apply conv3 to the output of conv2        
        x = self.conv3(x)

        # Reshape the output tensor to match the desired shape
        x = x.view(-1, 10)
        # Apply log softmax activation to the output
        return F.log_softmax(x, dim=1)

    # Define a method to display the summary of the model    
    def summary(self, input_size=None):
        return summary(self, input_size=input_size, col_names=["input_size", "output_size", "num_params", "params_percent"])
    
# Define a training function
def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    # Create a progress bar for training
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to the specified device (GPU)
        data, target = data.to(device), target.to(device)
        # Reset the gradients
        optimizer.zero_grad()
        # Perform forward pass
        output = model(data)
        # Compute the loss
        loss = F.nll_loss(output, target)
        # Perform backward pass to compute gradients
        loss.backward()
        # Update the model parameters
        optimizer.step()
        # Update the progress bar description
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')

# Define a testing function
def test(model, device, test_loader):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    # Disable gradient calculation for inference
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to the specified device (GPU)
            data, target = data.to(device), target.to(device)
            # Perform forward pass
            output = model(data)
            # Compute the loss, and sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # Count the correct predictions  
            correct += pred.eq(target.view_as(pred)).sum().item()
    # Average the test loss
    test_loss /= len(test_loader.dataset)
    # Print the test results
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))