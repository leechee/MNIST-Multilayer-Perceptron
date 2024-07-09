import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# set seed for repoducibility
torch.manual_seed(19)

# import data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1037),(0.3081))]) # normalize with MNIST mean and standard dev values found online

# gets training and testing data
mnist_train = datasets.MNIST(root= 'data', train = True, download = True, transform=transform)
mnist_test = datasets.MNIST(root= 'data', train = False, download = True, transform=transform)

# train_dataset, val_dataset = random_split(train_dataset, [train_size, validation_size])
mnist_train, mnist_val = random_split(mnist_train,[55000,5000])

val_loader = DataLoader(mnist_val, batch_size= 50, shuffle= False)
train_loader = DataLoader(mnist_train, batch_size= 100, shuffle= False)
test_loader = DataLoader(mnist_test, batch_size= 50, shuffle= False)

#making the model
class NeuralNetwork(nn.Module):
    def __init__ (self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(784,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,10)
        )

    def forward(self,x):
        x = x.view(-1, 784) # Flatten
        return self.layers(x)

#training the model
'''
for images, labels in train_loader
clear past gradients
foward pass
compute losses
backpropagation
update the weights
'''

def train(neural_network, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):

        neural_network.train() # sets into test mode

        # training set -----------------------------------------------------------------------------
        
        total_train_loss = 0
        correct_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad() # clears past gradients
            output = neural_network(images) # forward pass
            loss_train = criterion(output, labels) # compute losses
            loss_train.backward() # backpropagation
            optimizer.step() # updates the weights
            total_train_loss += loss_train.item()

            #computes each prediction, and then counts how many instances it got right
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(labels.view_as(pred)).sum().item()

        total_train_loss /= len(train_loader.dataset)


        # validation set ---------------------------------------------------------------------------

        neural_network.eval() # sets into eval mode

        total_validation_loss = 0
        correct_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                output = neural_network(images)
                loss_val = criterion(output, labels)
                total_validation_loss += loss_val.item()

                #computes each prediction, and then counts how many instances it got right
                pred = output.argmax(dim=1, keepdim=True)
                correct_val += pred.eq(labels.view_as(pred)).sum().item()

        total_validation_loss /= len(val_loader.dataset)

        #testing set--------------------------------------------------------------------------------

        total_test_loss = 0
        correct_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                output = neural_network(images)
                test_loss = criterion(output, labels)
                total_test_loss += test_loss.item()

                #computes each prediction, and then counts how many instances it got right
                pred = output.argmax(dim=1, keepdim=True)
                correct_test += pred.eq(labels.view_as(pred)).sum().item()

        total_test_loss /= len(test_loader.dataset)

        # print out losses at each epoch
        print(f"Epoch {epoch+1}/{num_epochs} " 
            f"Train Loss: {total_train_loss:.3f} "
            f"Validation Loss: {total_validation_loss:.3f} "
            f"Test Loss: {total_test_loss:.3f}"
            )
    
    #print out accuracies at the end
    print(f"Train Accuracy: {correct_train/len(train_loader.dataset)*100:.3f} " 
          f"Validation Accuracy: {correct_val/len(val_loader.dataset)*100:.3f} "
          f"Testing Accuracy: {correct_test/len(test_loader.dataset)*100:.3f}"
          )

#declare model
learning_rate = 0.001
num_epochs = 5
neural_network = NeuralNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neural_network.parameters(), learning_rate)

train(neural_network, criterion, optimizer, num_epochs)