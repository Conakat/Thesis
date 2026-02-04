import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report
from util import load_model
from CNN import CNN1, CNN2
import torch.nn as nn


def test_model(x_test, y_test, batch_size, channels, h, w, num_classes, model_path, name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data = torch.from_numpy(x_test).float() 
    test_labels = torch.from_numpy(y_test).long()
    test_data = test_data.view(-1, channels, h, w)
    print(test_data.shape)
    testing_dataset = TensorDataset(test_data, test_labels)
    testDataLoader = DataLoader(testing_dataset, batch_size=batch_size)

    if name == 'full-resolution':
        model = CNN2(channels, num_classes)
    else:
        model = CNN1(channels, num_classes)
        
    model = load_model(model, model_path)
    model.to(device)  # Move the model to the appropriate device

    criterion = nn.CrossEntropyLoss()
    classes = ["all finger r", "all finger f", "all finger e", "thumb f", "thumb e", "index f", "index e", "middle f", "middle e", "ring f", "ring e", "pingy f", "pingy e"]
    total_correct = 0
    total_loss = 0.0
    total_samples = 0
    # εφαρμογή του μοντέλου στο test set
    with torch.no_grad():
        model.eval()

        preds = []

        for image, label in testDataLoader:
            
            image = image.to(device)

            pred = model(image)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

            loss = criterion(pred, label)
            total_loss += loss.item() * image.size(0)  # Multiply by the number of samples in the batch

            # Calculate correct predictions
            total_correct += (pred.argmax(axis=1) == label).sum().item()
            total_samples += image.size(0)

        avg_test_loss = total_loss / total_samples
    # Calculate test accuracy
    test_accuracy = total_correct / total_samples

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} \n")
'''
    print("Unique classes in y_test:", np.unique(y_test))
    print("Unique classes in preds:", np.unique(np.array(preds)))

    for i in range(10):  # Check the first 10 examples
        print(f"Example {i+1}: Ground Truth - {classes[y_test[i]]}, Predicted - {classes[preds[i]]}")

    print(classification_report(y_test, np.array(preds), target_names=classes, zero_division=1))
'''