from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam        # TODO: Examine effects of different optim algos
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from pneumonia_dataset import load_datasets

import sys
import time
import threading

'''
Was bored waiting for model to train. Adds a spinner to the terminal output
so I know the model is still training.
'''
class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False

'''
List of sites used for template:
    https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212
    https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

#TODO: Heres a good source for some other model structure ideas: https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/#:~:text=1.,model%20to%20beat%20even%20today.
# Our conolution neural network
class Network(nn.Module):
    def __init__(self):
        print("Initialising model")
        super(Network, self).__init__()
        '''
        # TODO: Experiment with different network structures as different guides use different.
        # Initial is based on medium.com link above
        # This looks useful:
        # https://stats.stackexchange.com/questions/380996/convolutional-network-how-to-choose-output-channels-number-stride-and-padding/381032#381032
        '''
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
        self.fc3 = nn.Linear(84, 3)

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

def test_accuracy(model, data):
    outputs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data:
            output = model(images)
            #print(output.shape)
            _, predicted = torch.max(output.data, 1)
            #print(predicted)
            outputs += (predicted.to("cpu").tolist())
            all_labels += (labels.to("cpu").squeeze().tolist())

    print(outputs)
    print(labels)

    # Sklearn requires tensors to be on the cpu.
    return f1_score(all_labels, outputs, average="macro")

def train(epochs):
    # If possible we would like to use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"model is training using: {device}")

    train_data, test_data = load_datasets()
    train_dataloader = DataLoader(train_data, batch_size=500, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    print(len(train_data))

    model = Network()
    print(model)

    # TODO: Class weights since we have unbalanced dataset?
    # TODO: Higher learning rates could allow our model to converge with less epochs
    # could help reduce training time by using less epochs
    # Interestingly with a learning rate of 0.01 we see minimal to no changes to the loss function
    # with each iteration. So clearly we are converging very early but only achieve an f1 score of around
    # 0.77. After 350 iterations it actually started to imporove. Maybe a local minimum?
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_accuracy = 0

    model.to(device)

    for epoch in range(epochs):
        print(f"--------------- Epoch: {epoch + 1} ---------------")
        # data is list [(inputs, labels)]

        iterations_loss = []
        iterations_f1 = []
        losses = []
        f1_scores = []
        for i, (images, labels) in enumerate(train_dataloader):
            iterations_loss.append(i)    # TODO: Theres more efficient ways to do this. Could also do i + epoch * data size to combine all toghether in single graph

            # Make predictions then take a step
            optimizer.zero_grad()
            outputs = model(images)
            #print(f"outputs: {outputs}\nlabels: {labels}")
            loss = loss_fn(outputs, labels)
            #print(loss)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            #if i % 10 == 0:
                #print(f"Iteration {i} has accuracy: {test_accuracy(model, data)}")
        
            # Since a step may worsen accuracy our final step may not be the best
            # So we save the model that gave us the best result and use that as our
            # model
            iterations_f1.append(i)
            accuracy = 0
            with Spinner(): # I was bored
                accuracy = test_accuracy(model, test_dataloader)
            print(f"f1 score at iteration {i+1}: {accuracy}")
            f1_scores.append(accuracy)
            if accuracy > best_accuracy:
                save_model(model)
                best_accuracy = accuracy
        
        plt.plot(iterations_loss, losses)
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Loss for epoch " + str(epoch))
        plt.savefig("loss_score_epoch_" + str(epoch) + ".png")
        plt.show()

        plt.plot(iterations_f1, f1_scores)
        plt.xlabel("iteration")
        plt.ylabel("f1 score")
        plt.yscale("log")
        plt.title("F1 score for epoch " + str(epoch))
        plt.savefig("f1_score_epoch_" + str(epoch) + ".png")
        plt.show()
    
    with Spinner():
        accuracy = test_accuracy(model, test_dataloader)
        print(f"final f1 score: {accuracy}")

    # TODO: Load best model and return it
    # TODO: Create classification report with f1 score
    return model

if __name__ == "__main__":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    '''
    5 epochs and batch size 500
    -final f1 score: 0.9787454639709695
    Finished Training
    2889552.75
    '''

    model = train(1)

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training')
    print(start.elapsed_time(end))  # milliseconds
    # Then we test?