import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def CNN():      # CNN model
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(128 * 12 * 12, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model


def train_model(train_loader, model, criterion, optimizer, device):     # function to train the model
    model.train()                                                       # train mode
    train_loss = 0
    for inputs, labels in train_loader:                                 # iteration through all data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()                                           # resetting gradients
        outputs = model(inputs)                                         # passing inputs to model
        loss = criterion(outputs, labels.unsqueeze(1).float())          # counting loss
        loss.backward()                                                 # counting gradients
        optimizer.step()                                                # updating weights
        train_loss += loss.item()
    return train_loss / len(train_loader)


def evaluate_model(eval_loader, model, criterion, device):              # function to evaluate model on validation set
    model.eval()                                                        # evaluation mode
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():                                               # without computing gradients
        for inputs, labels in eval_loader:                              # iteration through all data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())      # counting loss
            val_loss += loss.item()
            predicted = torch.round(outputs)                            # for binary prediction
            total += labels.size(0)                                     # number of all samples
            correct += (predicted == labels.unsqueeze(1)).sum().item()  # number of correctly predicted labels
    
    accuracy = 100 * correct / total                                    # accuracy in %
    return accuracy, val_loss / len(eval_loader)


def test_model(test_loader, model, criterion, device):                  # function to test a model and plot confusion matrix
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs)
            all_labels.extend(labels.cpu().numpy())                     # adding labels to list
            all_predictions.extend(predicted.cpu().numpy())             # adding predictions to list
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
    # plotting onfusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


# plotting graphs
def plot_metrics(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

def main():
    batch_size = 32                  # defined hyperparameters
    learning_rate = 0.001
    num_epochs = 7                   # to reduce overfitting -- manual early stopping
    
    transform = transforms.Compose([ # data preprocessing
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    
    train_data = datasets.ImageFolder("full_dataset/training_set", transform=transform)  # creating data sets
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    valid_data = datasets.ImageFolder("full_dataset/validation_set", transform=transform)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    
    test_data = datasets.ImageFolder("full_dataset/test_set", transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           # if gpu available
    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # training the final model
    for epoch in range(num_epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        val_accuracy, val_loss = evaluate_model(valid_loader, model, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%")
    
    # plotting
    plot_metrics(train_losses, val_losses, val_accuracies)
    
    # testing the final model
    test_model(test_loader, model, criterion, device)
    
    # saving the final model
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Model saved successfully")

if __name__ == "__main__":
    main()