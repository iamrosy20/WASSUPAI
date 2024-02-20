# Design neural network with iris dataset

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Transform the data into pytorch tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Initialize dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# Define Neural Network Model with input layer, 2 hidden layers, and output layer
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)     # from input layer to hidden layer 1
        self.fc2 = nn.Linear(64, 128)   # from hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(128, 3)    # from hidden layer 2 to output layer
        self.relu = nn.ReLU()           # from hidden layer 2 to activation function ReLU

    def forward(self, x):
        x = self.relu(self.fc1(x))      # between hidden layer 1 and hidden layer 2
        x = self.relu(self.fc2(x))      # between hidden layer 2 and output layer
        x = self.fc3(x)                 # output layer
        return x

# Initialize model, loss function, optimizer
model = IrisNet()
criterion = nn.CrossEntropyLoss()   # automatically apply softmax function
optimizer = optim.Adam(model.parameters(), lr=0.001)
print('Bias of first hidden layer:', model.fc1.bias.data)       # check bias
print('Weight of first hidden layer:', model.fc1.weight.data)   # check weight

# Train the model
for epoch in range(100):
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()               # initialize gradient to 0
        outputs = model(inputs)             # forward propagation
        loss = criterion(outputs, labels)   # calculate loss
        loss.backward()                     # back propagation
        optimizer.step()                    # update parameter
        total_loss += loss.item()           # add current loss

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

# Evaluate the model
model.eval()            # model evaluation mode
with torch.no_grad():   # no gradient calculation (no model update)
    total = 0           # number of total samples
    correct = 0         # number of correctly predicted samples
    for inputs, labels in test_loader:  # call inputs and labels in batches
        outputs = model(inputs)         # return prediction
        _, predicted = torch.max(outputs.data, 1)   # get index of class with highest probability
        # _: max value of each row/column
        # predicted: index of that max value (class)
        # 1: dimension of max value
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test set: %d %%' % (correct / total * 100))
