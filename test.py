import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def CNN():                              # CNN model
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


def test_model(test_loader, model, criterion, device):              # function to test a model and plot confusion matrix
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():                                           # without computing gradients
        for inputs, labels in test_loader:                          # iteration through all data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)                                 # passing inputs to model
            predicted = torch.round(outputs)
            all_labels.extend(labels.cpu().numpy())                 # adding labels to list
            all_predictions.extend(predicted.cpu().numpy())         # adding predictions to list
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")                 # accuracy in %
    
    # plotting confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def run_model_test(model_path, test_dir, device):
    model = CNN().to(device)                                        # loading the saved model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([                                # preprocessing data
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    test_data = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=32)

    criterion = nn.BCELoss()

    test_model(test_loader, model, criterion, device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # if gpu available
    model_path = "cnn_model.pth"                                            # saved model
    test_dir = "dataset/test_set"
    
    run_model_test(model_path, test_dir, device)
