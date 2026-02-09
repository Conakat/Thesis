import sys

from google.colab import drive
drive.mount('/content/drive')

import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

model_save_path = '/content/drive/MyDrive/protonet_epoch_{}.pt'
model_load_path = '/content/drive/MyDrive/protonet_epoch_{}.pt'

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def extract_sample(n_ways, n_support, n_query, datax, data_reg, datay, batch_size):

    # Decimation based on Angle bins

    tasks = []
    query_classes = []
    for key in datax.keys():
      combined = list(zip(datax[key], data_reg[key]))
      np.random.shuffle(combined)
      datax[key], data_reg[key] = zip(*combined)

    for _ in range(batch_size):
        sample = []
        task_query_classes = []
        # Randomly select 'n_ways' classes for this task
        selected_classes = np.random.choice(np.unique(datay), n_ways, replace=False)
        #print(selected_classes)
        for cls in selected_classes:
            datax_cls = datax[cls]  # This is (400, 8, 960) for the class
            datareg_cls = data_reg[cls]  # Regularization data (400,)
            length = len(datax_cls)  # Number of samples (should be 400)

            # Binning data according to reg_val_cls
            reg_val_cls = np.array([val // (1 / n_support) for val in datareg_cls])
            reg_val_cls = reg_val_cls.astype(int)

            class_support_samples = []
            class_query_samples = []

            # Support set selection using binning
            support_indices = []
            for i in range(n_support):
                bin_indices = np.where(reg_val_cls == i)[0]  # Get indices that belong to bin i
                if len(bin_indices) > 0:
                    # Randomly pick a support sample from the bin
                    support_idx = np.random.choice(bin_indices)
                    support_indices.append(support_idx)
                    class_support_samples.append(datax_cls[support_idx])
                else:
                    print(f"Warning: No samples available for bin {i}. Filling with a random sample.")
                    support_idx = np.random.randint(0, length)
                    support_indices.append(support_idx)
                    class_support_samples.append(datax_cls[support_idx])

            # Exclude support samples from query sample selection
            available_query_indices = list(set(range(length)) - set(support_indices))

            # Check if there are enough samples for the query set
            if len(available_query_indices) < n_query:
                print(f"Warning: Not enough samples for query set in class {cls}. Using all available samples.")
                n_query = len(available_query_indices)

            # Query set selection
            query_indices = np.random.choice(available_query_indices, n_query, replace=False)
            for query_idx in query_indices:
                class_query_samples.append(datax_cls[query_idx])
                task_query_classes.append(cls)
            # Combine support and query samples for this class
            class_samples = np.concatenate((class_support_samples, class_query_samples), axis=0)
            sample.append(class_samples)

        # Convert the task to a tensor and append to tasks list
        sample = np.array(sample)  # Shape: (n_ways, n_support + n_query, 8, 960)
        sample = torch.from_numpy(sample).float()
        tasks.append(sample)
        query_classes.append(task_query_classes)
    # Stack all tasks into a single tensor
    #print(query_classes)
    tasks = torch.stack(tasks)

    return {
        'images': tasks,
        'n_ways': n_ways,
        'n_support': n_support,
        'n_query': n_query,
        'query_classes': query_classes
    }



subject = {1:'A1',2:'A2',3:'A3',4:'A4'}
experiment = {2:'exp2'}
session = {1:'session1',2:'session2',3:'session3'}
gesture_dict = {1:'Rest',2:'FinePinch',3:'KeyGrip',4:'TripodGrip',5:'IndexPoint',6:'PowerGrip'}

def load_data(n_subject,n_experiment,n_session,data_type):

		#Save current working directory
		working_directory = os.getcwd()

		#Path were the training or testing folder is:
		if data_type == "Training":
			dataset_path = os.path.join(working_directory,'drive/MyDrive/{}/{}/{}/Dataset_CNN/Training'.format(subject[n_subject],experiment[n_experiment],session[n_session]))
			samples_per_class = 400
		else:
			dataset_path = os.path.join(working_directory,'drive/MyDrive/{}/{}/{}/Dataset_CNN/Testing'.format(subject[n_subject],experiment[n_experiment],session[n_session]))
			samples_per_class = 100

		#List containing the names of the different classes
		categories = os.listdir(dataset_path)

		#Images,Regression, and classificaiton matrix
		x = np.zeros((6*samples_per_class,8,960))
		y_regression = np.zeros((6*samples_per_class,1))
		y_classification = np.zeros((6*samples_per_class,1))

		for i,cat in enumerate(categories):

			#Path of each class
			category_path = os.path.join(dataset_path,cat)

			#Images
			if data_type == "Training":
				class_images = np.load(os.path.join(category_path,'train_data.npy'))
			else:
				class_images = np.load(os.path.join(category_path,'test_data.npy'))

			#Load the regression data
			class_regression = np.load(os.path.join(category_path,'regression_data.npy'))

			#Load the x data
			x[samples_per_class*i:samples_per_class*i+samples_per_class,:,:] = class_images[:,:,:]
			x = np.array(x)
			#Extract the regression targets from the class
			y_regression[samples_per_class*i:samples_per_class*i+samples_per_class,0] = class_regression

			#Assign the classification target
			y_classification[samples_per_class*i:samples_per_class*i+samples_per_class,0] = i

		x.astype(np.float32)
		y_regression.astype(np.float32)

		x_dict = {}
		for i, a_mode_data in enumerate(x, start=1):
			class_label = (i-1)//samples_per_class
			if class_label not in x_dict:
				x_dict[class_label] = []
			x_dict[class_label].append(a_mode_data)

		y_dict_classification = {}
		for i, classes in enumerate(y_classification, start=1):
			class_label = (i-1)//samples_per_class
			if class_label not in y_dict_classification:
				y_dict_classification[class_label] = []
			y_dict_classification[class_label].append(classes)

		y_dict_reg = {}
		for i, IMU_data in enumerate(y_regression, start=1):
			class_label = (i-1)//samples_per_class
			if class_label not in y_dict_reg:
				y_dict_reg[class_label] = []
			y_dict_reg[class_label].append(IMU_data)

		return x, x_dict, y_dict_reg, y_regression, y_classification, y_dict_classification

def split_test(x_test,y_test_classification,y_test_regression):

	#Number of samples per gesture
	n = 100

	#New arrays
	x_test_new = np.zeros((450,8,960))
	x_validation = np.zeros((150,8,960))
	y_test_reg_new = np.zeros((450,1))
	y_test_classification_new = np.zeros((450,1))

	y_validation_reg = np.zeros((150,1))
	y_validation_classification = np.zeros((150,1))

	for i in range(0,6):
		#New test samples
		x_test_new[i*75:i*75+75,:,:] = x_test[i*n+25:i*n+n,:,:]
		y_test_reg_new[i*75:i*75+75,0] = y_test_regression[i*n+25:i*n+n,0]
		y_test_classification_new[i*75:i*75+75,0] = y_test_classification[i*n+25:i*n+n,0]

		#New validation samples
		x_validation[i*25:i*25+25,:,:] =x_test[i*n:i*n+25,:,:]
		y_validation_reg[i*25:i*25+25,0] = y_test_regression[i*n:i*n+25,0]
		y_validation_classification[i*25:i*25+25,0] = y_test_classification[i*n:i*n+25,0]

	y_validation_reg = np.array(y_validation_reg)
	y_test_reg_new = np.array(y_test_reg_new)

	x_dict_validation = {}
	for i, a_mode_data in enumerate(x_validation, start=1):
			class_label = (i-1)//25
			if class_label not in x_dict_validation:
				x_dict_validation[class_label] = []
			x_dict_validation[class_label].append(a_mode_data)

	x_dict_test = {}
	for i, a_mode_data in enumerate(x_test_new, start=1):
			class_label = (i-1)//75
			if class_label not in x_dict_test:
				x_dict_test[class_label] = []
			x_dict_test[class_label].append(a_mode_data)

	y_dict_val_reg = {}
	for i, IMU_data in enumerate(y_validation_reg, start=1):
			class_label = (i-1)//25
			if class_label not in y_dict_val_reg:
				y_dict_val_reg[class_label] = []
			y_dict_val_reg[class_label].append(IMU_data)

	y_dict_test_reg = {}
	for i, IMU_data in enumerate(y_test_reg_new, start=1):
			class_label = (i-1)//75
			if class_label not in y_dict_test_reg:
				y_dict_test_reg[class_label] = []
			y_dict_test_reg[class_label].append(IMU_data)

	return x_test_new,x_dict_test,y_dict_test_reg,y_test_reg_new,y_test_classification_new,x_validation,x_dict_validation,y_dict_val_reg,y_validation_reg,y_validation_classification


def generate_data (subject, experiment, session):
  x_train, x_dict_train, y_dict_train_reg, y_train_reg ,y_train_class, y_dict_train_class = load_data(subject,experiment,session, 'Training')
  x_val, _, _, y_val_reg, y_val_class, _ = load_data(subject,experiment,session, 'Testing')
  x_test, x_dict_test,y_dict_test_reg, y_test_reg, y_test_class,_,_,_,_,_ = split_test(x_val, y_val_class, y_val_reg)
  _,_,_,_,_,x_val, x_dict_val, y_dict_val_reg, y_val_reg, y_val_class = split_test(x_val,y_val_class,y_val_reg)
  y_train_class = y_train_class.astype(int)
  y_val_class = y_val_class.astype(int)
  y_test_class = y_test_class.astype(int)

  return (x_train, x_dict_train, y_dict_train_reg, y_train_reg, y_train_class, y_dict_train_class,
  x_val, x_dict_val, y_dict_val_reg, y_val_reg, y_val_class,
  x_test, x_dict_test, y_dict_test_reg, y_test_reg, y_test_class
  )


x_train11, x_dict_train11, y_dict_train_reg11, y_train_reg11, y_train_class11, y_dict_train_class11, x_val11, x_dict_val11, y_dict_val_reg11, y_val_reg11, y_val_class11, x_test11, x_dict_test11, y_dict_test_reg11, y_test_reg11, y_test_class11 = generate_data(subject=1, experiment=2, session=1)
x_train12, x_dict_train12, y_dict_train_reg12, y_train_reg12, y_train_class12, y_dict_train_class12, x_val12, x_dict_val12, y_dict_val_reg12, y_val_reg12, y_val_class12, x_test12, x_dict_test12, y_dict_test_reg12, y_test_reg12, y_test_class12 = generate_data(subject=1, experiment=2, session=2)
x_train13, x_dict_train13, y_dict_train_reg13, y_train_reg13, y_train_class13, y_dict_train_class13, x_val13, x_dict_val13, y_dict_val_reg13, y_val_reg13, y_val_class13, x_test13, x_dict_test13, y_dict_test_reg13, y_test_reg13, y_test_class13 = generate_data(subject=1, experiment=2, session=3)

x_train21, x_dict_train21, y_dict_train_reg21, y_train_reg21, y_train_class21, y_dict_train_class21, x_val21, x_dict_val21, y_dict_val_reg21, y_val_reg21, y_val_class21, x_test21, x_dict_test21, y_dict_test_reg21, y_test_reg21, y_test_class21 = generate_data(subject=2, experiment=2, session=1)
x_train12, x_dict_train22, y_dict_train_reg22, y_train_reg22, y_train_class22, y_dict_train_class22, x_val22, x_dict_val22, y_dict_val_reg22, y_val_reg22, y_val_class22, x_test22, x_dict_test22, y_dict_test_reg22, y_test_reg22, y_test_class22 = generate_data(subject=2, experiment=2, session=2)
x_train13, x_dict_train23, y_dict_train_reg23, y_train_reg23, y_train_class23, y_dict_train_class23, x_val23, x_dict_val23, y_dict_val_reg23, y_val_reg23, y_val_class23, x_test23, x_dict_test23, y_dict_test_reg23, y_test_reg23, y_test_class23 = generate_data(subject=2, experiment=2, session=3)

x_train31, x_dict_train31, y_dict_train_reg31, y_train_reg31, y_train_class31, y_dict_train_class31, x_val31, x_dict_val31, y_dict_val_reg31, y_val_reg31, y_val_class31, x_test31, x_dict_test31, y_dict_test_reg31, y_test_reg31, y_test_class31 = generate_data(subject=3, experiment=2, session=1)
x_train32, x_dict_train32, y_dict_train_reg32, y_train_reg32, y_train_class32, y_dict_train_class32, x_val32, x_dict_val32, y_dict_val_reg32, y_val_reg32, y_val_class32, x_test32, x_dict_test32, y_dict_test_reg32, y_test_reg32, y_test_class32 = generate_data(subject=3, experiment=2, session=2)
x_train33, x_dict_train33, y_dict_train_reg33, y_train_reg33, y_train_class33, y_dict_train_class33, x_val33, x_dict_val33, y_dict_val_reg33, y_val_reg33, y_val_class33, x_test33, x_dict_test33, y_dict_test_reg33, y_test_reg33, y_test_class33 = generate_data(subject=3, experiment=2, session=3)

x_train41, x_dict_train41, y_dict_train_reg41, y_train_reg41, y_train_class41, y_dict_train_class41, x_val41, x_dict_val41, y_dict_val_reg41, y_val_reg41, y_val_class41, x_test41, x_dict_test41, y_dict_test_reg41, y_test_reg41, y_test_class41 = generate_data(subject=4, experiment=2, session=1)
x_train42, x_dict_train42, y_dict_train_reg42, y_train_reg42, y_train_class42, y_dict_train_class42, x_val42, x_dict_val42, y_dict_val_reg42, y_val_reg42, y_val_class42, x_test42, x_dict_test42, y_dict_test_reg42, y_test_reg42, y_test_class42 = generate_data(subject=4, experiment=2, session=2)
x_train43, x_dict_train43, y_dict_train_reg43, y_train_reg43, y_train_class43, y_dict_train_class43, x_val43, x_dict_val43, y_dict_val_reg43, y_val_reg43, y_val_class43, x_test43, x_dict_test43, y_dict_test_reg43, y_test_reg43, y_test_class43 = generate_data(subject=4, experiment=2, session=3)

train_gestures = [5, 0]  # first 3 gestures
val_gestures = [1, 3]  # remaining 3 gestures
test_gestures = [2, 4]

N = 50

x_test_dicts = [
    x_dict_test11, x_dict_test12, x_dict_test13,
    x_dict_test21, x_dict_test22, x_dict_test23,
    x_dict_test31, x_dict_test32, x_dict_test33,
    x_dict_test41, x_dict_test42, x_dict_test43
]


for d in x_test_dicts:
    for g in val_gestures:
        d[g] = d[g][:N]

y_train_class = train_gestures
y_val_class = val_gestures
y_test_class = test_gestures

def split_and_combine(x_dicts, y_dicts, selected_gestures):
    """
    x_dicts: list of x dictionaries (one per subject/session)
    y_dicts: list of y dictionaries (same order)
    selected_gestures: list of gesture IDs

    Returns:
        combined_x, combined_y
    """

    combined_x = {}
    combined_y = {}

    for g in selected_gestures:
        xs = []
        ys = []

        for xd, yd in zip(x_dicts, y_dicts):
            xs.append(xd[g])
            ys.append(yd[g])

        combined_x[g] = np.concatenate(xs, axis=0)
        combined_y[g] = np.concatenate(ys, axis=0)

    return combined_x, combined_y


x_train_dicts = [x_dict_train13, x_dict_train23, x_dict_train33, x_dict_train43]
y_train_dicts = [y_dict_train_reg13, y_dict_train_reg23, y_dict_train_reg33, y_dict_train_reg43]

x_train_split, y_train_reg_split = split_and_combine(
    x_train_dicts,
    y_train_dicts,
    train_gestures
)

x_val_dicts = [x_dict_test13, x_dict_test23, x_dict_test33, x_dict_test43]
y_val_dicts = [y_dict_test_reg13, y_dict_test_reg23, y_dict_test_reg33, y_dict_test_reg43]

x_val_split, y_val_reg_split = split_and_combine(
    x_val_dicts,
    y_val_dicts,
    val_gestures
)

x_test_dicts = [x_dict_test13, x_dict_test23, x_dict_test33, x_dict_test43]
y_test_dicts = [y_dict_test_reg13, y_dict_test_reg23, y_dict_test_reg33, y_dict_test_reg43]

x_test_split, y_test_reg_split = split_and_combine(
    x_test_dicts,
    y_test_dicts,
    test_gestures
)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def load_model():

    def conv_block(kernel, dropout_rate, size):

        return nn.Sequential(
        nn.Conv1d(in_channels=size,out_channels=size,kernel_size=kernel,stride=1,padding=0),
        nn.BatchNorm1d(size),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=5,stride=5),
        nn.Dropout(dropout_rate)
    )

    kernels = [51,23,8,4]
    dropout_rate = 0.1
    size = 32

    encoder = nn.Sequential(
        nn.Conv1d(in_channels=8,out_channels=size,kernel_size=kernels[0],stride=1,padding=0),
        nn.BatchNorm1d(size),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=5,stride=5),
        nn.Dropout(dropout_rate),

        conv_block(kernels[1], dropout_rate, size),

        conv_block(kernels[2], dropout_rate, size),

        nn.Conv1d(in_channels=size,out_channels=size,kernel_size=kernels[3],stride=1,padding=0),
        nn.BatchNorm1d(size),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2,stride=2),
        nn.Dropout(dropout_rate),

        Flatten()
    )

    return ProtoNet(encoder)

class ProtoNet(nn.Module):

    def __init__(self, encoder):

       super(ProtoNet, self).__init__()
       self.encoder = encoder

    def set_forward_loss(self, sample):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sample_images = sample['images'].to(device)
        query_classes = sample['query_classes']
        n_ways = sample['n_ways']
        n_support = sample['n_support']
        n_query = sample['n_query']
        #print(query_classes)
        batch_size = sample_images.size(0)
        #print(batch_size)

        x_support = sample_images[:, :, :n_support]
        x_query = sample_images[:, :, n_support:]
        #print("x_support size:", x_support.size())
        #print("x_query size:", x_query.size())

        x_support = x_support.contiguous().view(batch_size, n_ways * n_support, *x_support.size()[3:])
        x_query = x_query.contiguous().view(batch_size, n_ways * n_query, *x_query.size()[3:])

        
        #print("x_support size:", x_support.size())
        #print("x_query size:", x_query.size())
  
        # Assign the true class labels to target_inds
        query_classes = torch.tensor(query_classes).view(batch_size, n_ways*n_query, 1).long().to(device)


        target_inds = torch.arange(0, n_ways).view(n_ways, 1, 1).expand(n_ways, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.to(device)
        target_inds = target_inds.unsqueeze(0).expand(batch_size, -1, -1, -1)  # shape: (batch_size, n_ways, n_query,1)

        # for example n_way=6, n_query=1 and n_support=5
        x_support_flat = x_support.view(-1, *x_support.size()[2:])  # [16 * (6*5), 8, 960]
        x_query_flat = x_query.view(-1, *x_query.size()[2:])  # [16 * (6*1), 8, 960]

        # Concatenate along the batch dimension
        x = torch.cat([x_support_flat, x_query_flat], 0)  # [16 * (30 + 6), 8, 960]
       
        z = self.encoder.forward(x)

        z_dim = z.size(-1)
        z_proto = z[:batch_size * n_ways * n_support].view(batch_size, n_ways, n_support, z_dim).mean(2)
        z_query = z[batch_size * n_ways * n_support:].view(batch_size, n_ways * n_query, z_dim)
        
        dists = euclidean_dist(z_query, z_proto)
        log_p_y = F.log_softmax(-dists, dim=2).view(batch_size, n_ways, n_query, -1)

        loss_val = -log_p_y.gather(3, target_inds).squeeze().view(batch_size, -1).mean(1).mean()
        _, y_hat = log_p_y.max(3)
        y_hat_replaced = torch.zeros_like(y_hat)
        for i in range(batch_size):
            for j in range(n_ways*n_query):
                y_hat_replaced[i][j] = query_classes[i][y_hat[i, j]]
        
        acc_val = torch.eq(y_hat_replaced.squeeze(), query_classes.squeeze()).float().mean()
        
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat_replaced
            }


    def euclidean_dist(x, y):

        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        assert d == y.size(2)

        x = x.unsqueeze(2).expand(x.size(0), n, m, d)
        y = y.unsqueeze(1).expand(y.size(0), n, m, d)

        return torch.pow(x - y, 2).sum(3)


def train(model, optimizer, train_x, train_reg, train_y, val_x, val_reg, val_y, n_way, n_support, n_query, max_epoch, epoch_size, model_save_path, patience, min_delta, batch_size):

    epoch = 0 #epochs done so far
    stop = False #status to know when to stop
    epoch_loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    model.train()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0
        val_running_loss = 0.0
        val_running_acc = 0.0
        for episode in range(epoch_size):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_reg, train_y, batch_size)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']
            #print(output['y_hat'])
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), model_save_path.format(epoch+1))
        print('Epoch {:d} trained and model parameters saved.'.format(epoch+1))

        # Validation
        model.eval()
        model.load_state_dict(torch.load(model_save_path.format(epoch+1), weights_only=True))
        epoch_size_val = 100
        with torch.no_grad():
            for episode in range(epoch_size_val):  # You might want to use a smaller epoch size for validation
                sample = extract_sample(n_way, n_support, n_query, val_x, val_reg, val_y, batch_size)
                loss, output = model.set_forward_loss(sample)
                val_running_loss += output['loss']
                val_running_acc += output['acc']
                #print(output['y_hat'])
        epoch_loss = running_loss / epoch_size
        epoch_loss_list.append(epoch_loss)
        epoch_acc = running_acc / epoch_size
        acc_list.append(epoch_acc)

        # Validation metrics
        val_epoch_loss = val_running_loss / epoch_size_val
        val_loss_list.append(val_epoch_loss)
        val_epoch_acc = val_running_acc / epoch_size_val
        val_acc_list.append(val_epoch_acc)

        print(f'Epoch {epoch+1} -- Training Loss: {epoch_loss:.4f} Training Acc: {epoch_acc:.4f} -- Validation Loss: {val_epoch_loss:.4f} Validation Acc: {val_epoch_acc:.4f}')

        # Early stopping
        if val_epoch_loss < best_val_loss - min_delta:
            best_val_loss = val_epoch_loss
            best_val_acc = val_epoch_acc
            patience_counter = 0  # reset the counter if we get a better validation metric
        else:
            patience_counter += 1  # increment the counter if no improvement
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                stop = True

        epoch += 1
        scheduler.step()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch+1), epoch_loss_list, label='Training Loss')
    plt.plot(range(1, epoch+1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epochs')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch+1), acc_list, label='Training Accuracy')
    plt.plot(range(1, epoch+1), val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs. Epochs')
    plt.legend()
    plt.show()

def test(model, test_x, test_reg, test_y, n_ways, n_support, n_query, test_episode, model_path, batch_size):
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_acc = 0.0

    accuracies = []

    for episode in range(test_episode):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_reg, test_y, batch_size)
        loss, output = model.set_forward_loss(sample)
        running_loss += output['loss']
        running_acc += output['acc']
        #print(output['y_hat'])

        accuracies.append(output['acc'])
    print(accuracies)
    accuracies = np.array(accuracies)

    mean_accuracy = np.mean(accuracies) * 100
    std_accuracy = np.std(accuracies)

    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
    print('Test results -- Mean Accuracy: {:.4f} ± {:.4f}'.format(mean_accuracy, std_accuracy))

n_way = 2
n_support = 5
n_query = 1

batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model()
model = model.to(device)

patience = 6
min_delta = 0.02
optimizer = optim.Adam(model.parameters(),lr=0.1e-2, weight_decay=1e-4)
#model_save_path = "model_epoch_{}.pth"
max_epoch = 15
epoch_size = 200

train(model, optimizer, x_train_split, y_train_reg_split, y_train_class, x_val_split, y_val_reg_split, y_val_class, n_way, n_support, n_query, max_epoch, epoch_size, model_save_path, patience, min_delta, batch_size)
test_episode = 100
test(model, x_test_split, y_test_reg_split, y_test_class, n_way, n_support, n_query, test_episode, model_load_path.format(3), batch_size)
