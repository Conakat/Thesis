import create_Dataset
import constants
import split_for_CNN
import train
import testing_phase

# Links που με βοήθησαν για να φτιάξω το πρώτο μου νευρωνικό δίκτυο + chat.gpt 
# https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
# https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/

# παίρνω τα x_train, x_test μέσω του module split_for_CNN

channels = [1, 3]
dataset_id = [1, 2]


x_train, x_test = split_for_CNN.load_maps(constants.video_directory2, constants.sorted_video_files2, constants.ds2_train_path_fs, constants.ds2_test_path_fs, 'full-resolution', dataset_id[1])
#x_train, x_test = split_for_CNN.load_maps(constants.video_directory1, constants.sorted_video_files1, constants.path_for_kNN1_BD_train, constants.path_for_kNN1_BD_test, 'full-resolutio', dataset_id[0])
#x_train, x_test = split_for_CNN.load_optflow(constants.video_directory2, constants.sorted_video_files2, constants.training_path_opt2_15_med, constants.testing_path_opt2_15_med, 15,  dataset_id[1])
classes = ["all finger r", "all finger f", "all finger e", "thumb f", "thumb e", "index f", "index e", "middle f", "middle e", "ring f", "ring e", "pingy f", "pingy e"]

# λαμβάνω τα labels του DS1 και τα y_train, y_test περιλαμβάνουν 104(8 per gesture) και 26(2 per gesture) labels αντίστοιχα

y = create_Dataset.labels_DS2()
y_train, y_test = create_Dataset.labels_SVM(y)

height, width = x_train.shape[1], x_train.shape[2]
#height, width = x_train.shape[2], x_train.shape[3]
print(height, width)
print(y_train, y_test)

print(x_train.shape, x_test.shape)

# δημιουργώ τα training kai validation data για το model καθώς και τα αντίστοιχα labels.
sample_ratio = 0.5

test_data, test_labels, val_data, val_labels = split_for_CNN.train_val_split2(sample_ratio, x_test, y_test)

print(test_data.shape, test_labels, val_data.shape, val_labels)

# change to channels depending on the type of activity map

# ορισμός παραμέτρων για τον optimizer 
batch_size = 8 # to 10 kalo vriskei th mia xeironomia ring-e me lr=0.0001
num_classes = 13
learning_rate = 0.001
epochs = 8

model_path = "trained_model.pth"

train.train_CNN(x_train, y_train, val_data, val_labels, 1, height, width, learning_rate, batch_size, num_classes, epochs, model_path, 'full-resolution')
testing_phase.test_model(test_data, test_labels, batch_size, 1, height, width, num_classes, model_path, 'full-resolution')














'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# κανονικοποίηση των δεδομένων στο εύρος [0,1] 
train_data = torch.from_numpy(train_data).float()/255.0
train_labels = torch.from_numpy(train_labels).long()
print(train_labels)

val_data = torch.from_numpy(val_data).float()/255.0
val_labels = torch.from_numpy(val_labels).long()

test_data = torch.from_numpy(x_test).float()/255.0
test_labels = torch.from_numpy(y_test).long()
print(test_labels)

# reshape the tensor to have the dimensions expected by the neural network model
train_data = train_data.view(-1, num_channels, height, width)
print(train_data.shape)

val_data = val_data.view(-1, num_channels, height, width)
print(val_data.shape)

test_data = test_data.view(-1, num_channels, height, width)
print(test_data.shape)

# convenient way to combine input data (features) and corresponding labels into a single dataset
training_dataset = TensorDataset(train_data, train_labels)
validation_dataset = TensorDataset(val_data, val_labels)
testing_dataset = TensorDataset(test_data, test_labels)

# CNN model 1ος τρόπος

class CNN(nn.Module):
    def __init__(self, num_channels, num_classes, dropout_rate = 0.5):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.utils.parametrizations.weight_norm(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.utils.parametrizations.weight_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.utils.parametrizations.weight_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
            
        
            nn.utils.parametrizations.weight_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        

        # τα 76, 42 προέκυψαν από λόγω της χρήσης του window size του max pooling που είναι 2x2
        # αρχικα έχω διαστάσεις 610x342, με το πρώτο maxpooling layer γίνεται
        # 305x171 --> 2o δίνει 152x85 --> 3ο δίνει 76x42 

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*38*21, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)

        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 2ος τρόπος χωρίς τη χρήση Sequential
class CNN(nn.Module):
    def __init__(self, num_channels, num_classes, dropout_rate = 0.5):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*76*42, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.dropout(out)
        output = self.fc2(out)
        
        return output

'''