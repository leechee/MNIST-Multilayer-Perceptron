# MNIST-Multilayer-Perceptron
![[mnist]](assets/mnist.png)


The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. It contains 60,000 training images and 10,000 testing images. I am still fixing the validation set for this model.

## Objective

In this repository, I coded a deep neural network with a multilayer perceptron. The model has two hidden layers, the first with 256 neurons and the second with 128. The activation function is ReLU, and PyTorch.nn are implemented.

## Getting Started
### Python Environment
Download and install Python 3.8 or higher from the [official Python website](https://www.python.org/downloads/)

Optional, but I would recommend creating a venv. For Windows installation:
```
py -m venv .venv
.venv\Scripts\activate
```
For Unix/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```

Now install the necessary AI stack in the venv terminal. These libraries will aid with computational coding, data visualization, accuracy reports, preprocessing, etc. I used pip for this project.
```
pip install numPy
pip install matplotlib
```

For Torch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You will also need to install the torchvision MNIST dataset, which will be prompted in the terminal when called upon.

### Data Input
To input data from the MNIST data set, use the Torchvision library. Below is the code that transforms and splits the data into three sets of loaders. The validiation set, training set, and testing set. The validition set provides an unbiased evaluation of the model. 

```
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
```

### Results
![[results]](assets/fig1.png)

Results after 10 epochs: 

Training Accuracy: 99.376%
Testing Accuracy: 97.870%
Validation Accuracy: 97.420%
