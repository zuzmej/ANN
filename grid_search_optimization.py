import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):                                           # CNN model
    def __init__(self, filters1=32, filters2=64, filters3=128, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, filters1, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        self.input_size = self._get_conv_output((3, 100, 100))  # input image size is 100x100
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _get_conv_output(self, shape):                          # auxiliary function
        dummy_input = torch.rand(1, *shape)                     # generate a tensor with the shape of the input image
        x = self.pool(nn.ReLU()(self.conv1(dummy_input)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        return x.numel()
        
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def train_model(train_loader, model, criterion, optimizer, device, verbose=1):      # function to train the model
    model.train()                                                                   # train mode
    for inputs, labels in train_loader:                                             # iteration through all data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()                                                       # resetting gradients
        outputs = model(inputs)                                                     # passing inputs to model
        loss = criterion(outputs, labels.unsqueeze(1).float())                      # counting loss
        loss.backward()                                                             # counting gradients
        optimizer.step()                                                            # updating weights
    if verbose > 0:
        print("Training step completed.")


def evaluate_model(eval_loader, model, criterion, device, verbose=1):               # function to evaluate model on validation set
    model.eval()                                                                    # evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():                                                           # without computing gradients
        for inputs, labels in eval_loader:                                          # iteration through all data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs)                                        # for binary prediction
            total += labels.size(0)                                                 # number of all samples
            correct += (predicted == labels.unsqueeze(1)).sum().item()              # number of correctly predicted labels
    accuracy = 100 * correct / total                                                # accuracy in %
    if verbose > 0:
        print(f"Evaluation step completed. Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    batch_size = 32                     # defined hyperparameters
    learning_rate = 0.001
    num_epochs = 10
    verbose = 1                         # to enable detailed logging

    transform = transforms.Compose([    # data preprocessing
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder("dataset/training_set", transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    valid_data = datasets.ImageFolder("dataset/validation_set", transform=transform)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    
    test_data = datasets.ImageFolder("dataset/test_set", transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # if gpu available
    

    param_grid = {                             # parameters to be checked in grid search
        'filters1': [32, 64],
        'filters2': [64, 128],
        'filters3': [128, 256],
        'kernel_size': [3, 5],
        'batch_size': [32, 64],
        'learning_rate': [0.001, 0.0001]
    }
    
    best_params = None
    best_score = 0
    
    for filters1 in param_grid['filters1']:     # grid search
        for filters2 in param_grid['filters2']:
            for filters3 in param_grid['filters3']:
                for kernel_size in param_grid['kernel_size']:
                    for batch_size in param_grid['batch_size']:
                        for learning_rate in param_grid['learning_rate']:
                            model = CNN(filters1, filters2, filters3, kernel_size).to(device)
                            criterion = nn.BCELoss()
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                            valid_loader = DataLoader(valid_data, batch_size=batch_size)
                            
                            for epoch in range(num_epochs):
                                train_model(train_loader, model, criterion, optimizer, device, verbose)
                                if verbose > 0:
                                    print(f"Epoch [{epoch+1}/{num_epochs}] completed for filters1={filters1}, filters2={filters2}, filters3={filters3}, kernel_size={kernel_size}, batch_size={batch_size}, learning_rate={learning_rate}")
                            
                            accuracy = evaluate_model(valid_loader, model, criterion, device, verbose)
                            
                            if accuracy > best_score:
                                best_score = accuracy
                                best_params = {
                                    'filters1': filters1,
                                    'filters2': filters2,
                                    'filters3': filters3,
                                    'kernel_size': kernel_size,
                                    'batch_size': batch_size,
                                    'learning_rate': learning_rate
                                }
    print("Best Score: ", best_score)
    print("Best Params: ", best_params)
    

    # training the final model with best parameters
    model = CNN(best_params['filters1'], best_params['filters2'], best_params['filters3'], best_params['kernel_size']).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)
    
    for epoch in range(num_epochs):
        train_model(train_loader, model, criterion, optimizer, device, verbose)
        if verbose > 0:
            print(f"Final training: Epoch [{epoch+1}/{num_epochs}] completed.")
    

    # testing the final model
    test_accuracy = evaluate_model(test_loader, model, criterion, device, verbose)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    

    # saving the final model
    torch.save(model.state_dict(), "cnn_model_best.pth")
    print("Model saved successfully")


if __name__ == "__main__":
    main()