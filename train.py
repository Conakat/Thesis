import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib
import matplotlib.pyplot as plt
import CNN
from util import save_model

def train_CNN(train_data, train_labels, val_data, val_labels, channels, h, w, learning_rate, batch_size, num_classes, epochs, model_path, name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = torch.from_numpy(train_data).float() 
    train_data = train_data.view(-1, channels, h, w)
    print(train_data.shape)
    train_labels = torch.from_numpy(train_labels).long()
    #print(train_labels)

    val_data = torch.from_numpy(val_data).float() 
    val_labels = torch.from_numpy(val_labels).long()

    # reshape the tensor to have the dimensions expected by the neural network model
    
    val_data = val_data.view(-1, channels, h, w)
    print(val_data.shape)

    # convenient way to combine input data (features) and corresponding labels into a single dataset
    training_dataset = TensorDataset(train_data, train_labels)
    validation_dataset = TensorDataset(val_data, val_labels)
    
    criterion = nn.CrossEntropyLoss()

    if name == 'full-resolution':
        model = CNN.CNN2(channels, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    else:
        model = CNN.CNN1(channels, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  

    trainDataLoader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    valDataLoader = DataLoader(validation_dataset, batch_size=batch_size)

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Initialization process of the CNN model
    trainSteps = len(trainDataLoader.dataset) // batch_size
    valSteps = len(valDataLoader.dataset) // batch_size
    startTime = time.time()

    l1_lamda = 0.0015

    for e in range(0, epochs):
        model.train()

        totalTrainLoss = 0
        totalValLoss = 0

        trainCorrect = 0
        valCorrect = 0

        for image, label in trainDataLoader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, label)

            if name != "full-resolution":
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lamda * l1_norm 
            
            loss.backward()
            optimizer.step()

            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1)==label).type(torch.float).sum().item()

        # εφαρμογή του μοντέλου στο validation set 
        with torch.no_grad():
            model.eval()

            for image, label in valDataLoader:
                image = image.to(device)
                label = label.to(device)
                
                pred = model(image)
                totalValLoss += criterion(pred, label)

                valCorrect += (pred.argmax(1)==label).type(torch.float).sum().item()
        

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    
    plt.show()
    
    save_model(model, model_path)