from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam        # TODO: Examine effects of different optim algos
from torch.autograd import Variable

from pre_nn import load_data
from pre2 import load_test_data, load_train_data


'''
List of sites used for template:
    https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212
    https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

# Our conolution neural network
class Network(nn.Module):
    def __init__(self):
        print("Initialising model")
        super(Network, self).__init__()

        # TODO: Define layers and perceptrons here
        # TODO: Experiment with different network structures as different guides use different.
        # Initial is based on medium.com link above
        # This looks useful:
        # https://stats.stackexchange.com/questions/380996/convolutional-network-how-to-choose-output-channels-number-stride-and-padding/381032#381032
        
        '''
        5x5 square convolution kernel
        nn.Conv2d(in_channels = 2, out_channels = 6 TODO, kernel_size = 5)
        - in_channels
            - 1 since our image is greyscale
        - out_channels
            - TODO: To be checked. Use stackexchange link above
        - kernel_size
            - TODO: To be checked. Use stackexchange link above
        '''
        self.conv1 = nn.Conv2d(1, 6, 5) # Optional: stride, padding
        # Pool over a (2, 2) window. Image goes to dimensionality passed in
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers
        self.fc1 = nn.Linear(12544, 120)    # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)    # TODO: Is 10 the number of possible output labels?

    # Another interesting exercise could be to try other activation functions besides RELU
    def forward(self, x):
        #print(f"x is: {x}")
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12544)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def save_model(model):
    path = "./saved_model.pth"
    torch.save(model.state_dict(), path)

def load_model():
    path = "./saved_model.pth"
    model = Network()
    model.load_state_dict(torch.load(path))

    return model

# TODO: Change to f1 score
def test_accuracy(model, data):
    # TODO: If we're checking against test data we have to apply any preprocessing?
    # This could be done in get data
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for elem in data:
            images, labels = elem
            images = Variable(torch.tensor([images]).float().to(device))
            labels = Variable(torch.tensor([labels]).to(device))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def train(epochs):
    data = list(load_train_data())   # TODO. TODO: Since we're working with labels we may not need x_train, ... etc?
    model = Network()

    # Sanity check for data format
    print(list(data)[50])

    # TODO: Class weights since we have unbalanced dataset
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_accuracy = 0

    # If possible we would like to use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"model is training using: {device}")
    # Ensures the model is being trained on correct device. i.e. CPU or CUDA
    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        # data is list [(inputs, labels)]
        for i, (images, labels) in enumerate(data):
            #print(i)
            #print(f"image: {images}\nlabels: {labels}") # Debug
            # First we require the data to be stored in a tensor
            # .to(device) then ensures the tensor is stored on the correct device
            # i.e. CPU or CUDA
            # Variable provides a wrapper to represent a tensor as a node in a graph
            # We add images to a list of length 1 since conv2d requires more than 2 dimensions
            # TODO: We can use this to process multiple images in batches i.e. [image, image]
            images = Variable(torch.tensor([images]).float().to(device))
            labels = Variable(torch.tensor([labels]).to(device ))

            # Make predictions then take a step
            optimizer.zero_grad()
            outputs = model(images)
            #print(f"outputs: {outputs}\nlabels: {labels}")
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Iteration {i} has accuracy: {test_accuracy(model, data)}")
        
            # Since a step may worsen accuracy our final step may not be the best
            # So we save the model that gave us the best result and use that as our
            # model
            accuracy = test_accuracy(model, list(load_test_data()))
            if accuracy > best_accuracy:
                save_model(model)
                best_accuracy = accuracy
    
    # TODO: Load best model and return it
    return model

if __name__ == "__main__":
    model = train(5)

    # Then we test?