import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 1e-3

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


net = Net(input_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Train the Model

def train(model, train_loader, num_epochs):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            out = net(images)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()


# Test the Model
def evaluate(model, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28*28)

        total += labels.size(0)
        outputs = model(images)
        pred = outputs.data.max(1)[1]
        corrects = (outputs.data.max(1)[1] == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

train(net, train_loader, num_epochs)
evaluate(net, test_loader)

# Save the Model
torch.save(net.state_dict(), 'model.pkl')

#%%
