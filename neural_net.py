import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, AdamW, NAdam
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

# Our conolution neural network
class Network(nn.Module):
    def __init__(self):
        print("Initialising model")
        super(Network, self).__init__()

        '''
        5x5 square convolution kernel
        nn.Conv2d(in_channels = 2, out_channels = 6, kernel_size = 5)
        - in_channels
            - 1 since our image is greyscale
        - out_channels
            - From tutorial
        - kernel_size
            - From tutorial
        '''
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Pool over a (2, 2) window. Image goes to dimensionality passed in
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers
        self.fc1 = nn.Linear(12544, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # Perform convolutions and pooling
        # Reduces the dimensionality of the image(s)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Since nn.Linear requires a column vector we flatten the tensor
        x = x.view(-1, 12544)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

'''
Takes a CNN and saves its parameters so it can be loaded later
'''
def save_model(model):
    path = "./saved_model.pth"
    torch.save(model.state_dict(), path)

'''
Loads the parameters of a model
'''
def load_model():
    path = "./saved_model.pth"
    model = Network()
    model.load_state_dict(torch.load(path))

    return model

'''
Computes the predictions for some images in data and calculates the f1 score
'''
def calc_f1(model, data):
    outputs = []
    all_labels = []
    with torch.no_grad():
        # Make predictions
        for images, labels in data:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            outputs += (predicted.to("cpu").tolist())
            all_labels += (labels.to("cpu").squeeze().tolist())

    # Sklearn requires tensors to be on the cpu.
    return f1_score(all_labels, outputs, average="macro")

'''
Creates an instance of the model and then trains it.
'''
def train(epochs):
    # If possible we would like to use the GPU or MPS
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0") # GPU acceleration
    elif torch.backends.is_available():
        device =torch.device("mps")     # For Apple M1 chips
    print(f"model is training using: {device}")

    # Train and test dataloaders are iterables that return a tuple of the form ([images], [labels]) each iteration.
    train_data, test_data = load_datasets()
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    
    # Some initialisations
    fix, axs = plt.subplots(2, epochs, figsize=(80, 80))
    model = Network()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = NAdam(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_accuracy = 0

    # Models and tensors have to be sent to the GPU if we wish to utilise hardware acceleration
    model.to(device)

    for epoch in range(epochs):
        print(f"--------------- Epoch: {epoch + 1} ---------------")

        iterations_loss = []
        iterations_f1 = []
        losses = []
        f1_scores = []
        for i, (images, labels) in enumerate(train_dataloader):
            iterations_loss.append(i)    # TODO: Theres more efficient ways to do this. Could also do i + epoch * data size to combine all toghether in single graph

            # Make predictions then take a step
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            print(loss)
            losses.append(loss.cpu().detach().tolist())
            loss.backward()
            optimizer.step()

        
            # Since a step may worsen accuracy our final step may not be the best
            # So we save the model that gave us the best result and use that as our
            # model
            with Spinner(): # This takes a while so create a spinner so its clear its still running.
                accuracy = calc_f1(model, test_dataloader)
            # If this is the best accuracy so far save the weights.
            if accuracy > best_accuracy:
                save_model(model)
                best_accuracy = accuracy

            f1_scores.append(accuracy)
            iterations_f1.append(i)
            
            print(f"Iteration {i}:\nLoss: {loss}\nf1 score: {accuracy}\n\n")

        # Graph the results of the current epoch
        axs[0, epoch].step(iterations_loss, losses)
        axs[0, epoch].set_xlabel("iteration")
        axs[0, epoch].set_ylabel("Loss")
        axs[0, epoch].set_yscale("log")
        axs[0, epoch].set_title("Loss for epoch: " + str(epoch))

        axs[1, epoch].step(iterations_f1, f1_scores)
        axs[1, epoch].set_xlabel("iteration")
        axs[1, epoch].set_ylabel("f1 score")
        axs[1, epoch].set_title("F1 score for epoch " + str(epoch))

        # Since training takes so long we save a graph each epoch just in case it crashes.
        plt.savefig("f1_epoch_" + str(epoch) + "_.png")

    print(f"Finished training with f1_score: {best_accuracy}")
    plt.savefig("f1_score.png")
    plt.show()

    # We now load our best model and return it
    return load_model()

if __name__ == "__main__":
    model = train(10)