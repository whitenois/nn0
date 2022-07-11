import torch.utils.data
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
import random
import json

# Declaring important variables

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Declaring the train_loader
# Downoading the data for training

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

# Declaring the test_loader
# Dowloading the data for testing

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# Declaring the Net class

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Declaring the network

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# Load losses and counters for the graph

with open("./losses_and_counters.json", "r") as fp:
    lnc = json.load(fp)

train_losses = lnc["train_losses"]
train_counter = lnc["train_counter"]
test_losses = lnc["test_losses"]
test_counter = lnc["test_counter"]
n_epochs = lnc["n_epochs"]

network_state_dict = torch.load("./results/model.pth")
network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load("./results/optimizer.pth")
optimizer.load_state_dict(optimizer_state_dict)

def test_if_examples_loaded():
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = list(examples)[0]
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_for_epochs(n):
    global n_epochs
    for i in range(n_epochs + 1, n_epochs + n + 1):
        test_counter.append(i*len(train_loader.dataset))
        train(i)
        test()
    n_epochs += n

def show_result_graph(show_from=0):
    global n_epochs
    fig = plt.figure(figsize=(18, 12), dpi=400)
    plt.plot(train_counter[show_from * 94:], train_losses[show_from * 94:], color='blue')
    plt.scatter(test_counter[show_from:], test_losses[show_from:], color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    print(f"Epochs count: {n_epochs}")

def test_on_random():    
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = random.sample(list(examples), 1)[0]
    with torch.no_grad():
        output = network(example_data)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])

load_network_data()

# ----------------------------------------------------------------------------------- #
#                                FUNCTIONS TO USE                                     #
# ----------------------------------------------------------------------------------- #
#                                                                                     #
# - load_network_data()               Loads the network's model and optimizer from    #
#                                     ./results/model.pth and ./results.optimizer.pth #
#                                                                                     #
# - test_if_examples_loaded()         Show some examples from test_loader with ground #
#                                     truth shown                                     #
#                                                                                     #
# - train_for_epochs(n)               Train for some amount of new epochs             #
#                                                                                     #
# - show_result_graph()               Show the graph of training examples seen and    #
#                                     negative log likelihood loss, as long as the    #
#                                     number of epochs                                #
#                                                                                     #
# - test_on_random()                  Test the net on six random examples             #
#                                                                                     #
#-------------------------------------------------------------------------------------#
