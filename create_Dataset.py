import numpy as np #type:ignore
import os
import constants
import store_Dataset

def labels_DS1():

    num = 130   
    num_per_label = 10 
    num_labels = num // num_per_label 
    labels = np.repeat(np.arange(num_labels), num_per_label)

    return labels

def labels_DS2():

    num_per_label_rest = 25
    labels_rest = np.repeat(0, num_per_label_rest)
    label_10 = np.repeat(10, 29)
    labels_1t9 = np.repeat(np.arange(1,10), 30)
    labels_11t12 = np.repeat(np.arange(11,13), 30)
    labels = np.concatenate((labels_rest, labels_1t9, label_10, labels_11t12), axis = None)
    labels = labels.astype(int)

    return labels

def labels_SVM(labels):

    num = len(labels)

    if num==130:
        array = np.zeros((num, 2))
        for i in range(num):
            array[i][0] = labels[i]
            array[i][1] = constants.array1[i][2]
        
        y_train = array[array[:, 1] != False][:, 0].astype(int)
        y_test = array[array[:, 1] == False][:, 0].astype(int)
 
        return y_train, y_test
    
    else:
        array = np.zeros((num, 2))
        for i in range(num):
            array[i][0] = labels[i]
            array[i][1] = constants.array2[i][2]

        y_train = array[array[:, 1] != False][:, 0].astype(int)
        y_test = array[array[:, 1] == False][:, 0].astype(int)
        
        return y_train, y_test      
    

def optical_flow_files(train_path, test_path):

    x_train = []
    file_names = os.listdir(train_path)
    sorted_file_names = sorted(file_names, key=constants.custom_sort)

    for file_name in sorted_file_names:
         if file_name.endswith(".npy"):
            file_path = os.path.join(train_path, file_name)
            sample = np.load(file_path)
            x_train.append(sample)

    x_test = []
    file_names = os.listdir(test_path)
    sorted_file_names = sorted(file_names, key=constants.custom_sort)
    
    for file_name in sorted_file_names:
        if file_name.endswith(".npy"):
            file_path = os.path.join(test_path, file_name)
            sample = np.load(file_path)           
            x_test.append(sample)

    return np.array(x_train), np.array(x_test)


def activity_maps_files(path1, path2):

    x_train = []
    file_names = os.listdir(path1)
    sorted_file_names = sorted(file_names, key=constants.custom_sort)

    for file_name in sorted_file_names:
         if file_name.endswith(".npy"):
            file_path = os.path.join(path1, file_name)
            sample = np.load(file_path)
            x_train.append(sample)

    x_test = []
    file_names = os.listdir(path2)
    sorted_file_names = sorted(file_names, key=constants.custom_sort)

    for file_name in sorted_file_names:
         if file_name.endswith(".npy"):
            file_path = os.path.join(path2, file_name)
            sample = np.load(file_path)
            x_test.append(sample)

    return np.array(x_train), np.array(x_test)


def load_opticalFlow(video_directory, video_files, training_path, testing_path, median_filter_size, dataset_id):
    
    constants.create_directories1(training_path, testing_path)
    if dataset_id == 1:
        
        if (constants.is_directory_empty(training_path) and constants.is_directory_empty(testing_path)):
            store_Dataset.ds1_opticalFlow(video_directory, video_files, training_path, testing_path, median_filter_size)
            x_train, x_test = optical_flow_files(training_path, testing_path)
            print(f"Dataset {dataset_id}: Optical Flow with {median_filter_size}x{median_filter_size} - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
        
        else:
            x_train, x_test = optical_flow_files(training_path, testing_path)
            print(f"Dataset {dataset_id}: Optical Flow with {median_filter_size}x{median_filter_size} - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))

    else:
        if (constants.is_directory_empty(training_path) and constants.is_directory_empty(testing_path)):
            store_Dataset.ds2_opticalFlow(video_directory, video_files, training_path, testing_path, median_filter_size)
            x_train, x_test = optical_flow_files(training_path, testing_path)
            print(f"Dataset {dataset_id}: Optical Flow with {median_filter_size}x{median_filter_size} - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))

        else:
            x_train, x_test = optical_flow_files(training_path, testing_path)
            print(f"Dataset {dataset_id}: Optical Flow with {median_filter_size}x{median_filter_size} - Train shape:", np.shape(x_train), np.shape(x_test))

    return x_train, x_test
        
def load_activityMaps(video_directory, video_files, path1, path2, dataset_id):

    constants.create_directories1(path1, path2)
    if dataset_id == 1:

        if (constants.is_directory_empty(path1) and constants.is_directory_empty(path2)):
            store_Dataset.ds1_FSmaps(video_directory, video_files, path1, path2)
            x_train, x_test = activity_maps_files(path1, path2)
            print(f"Dataset {dataset_id}: Full Resolution Activity Maps - Dataset shape:", np.shape(x_train), np.shape(x_test))
        
        else:
            x_train, x_test = activity_maps_files(path1, path2)
            print(f"Dataset {dataset_id}: Full Resolution Activity Maps - Dataset shape:", np.shape(x_train), np.shape(x_test))

    else:

        if (constants.is_directory_empty(path1) and constants.is_directory_empty(path2)):
            store_Dataset.ds2_FSmaps(video_directory, video_files, path1, path2)
            x_train, x_test = activity_maps_files(path1, path2)
            print(f"Dataset {dataset_id}: Full Resolution Activity Maps - Dataset shape:", np.shape(x_train), np.shape(x_test))
        
        else:
            x_train, x_test = activity_maps_files(path1, path2)
            print(f"Dataset {dataset_id}: Full Resolution Activity Maps - Dataset shape:", np.shape(x_train), np.shape(x_test))

    return x_train, x_test

def load_BDmaps(video_directory, video_files, path1, path2, dataset_id):
    
    constants.create_directories1(path1, path2)
    if dataset_id == 1:

        if (constants.is_directory_empty(path1) and constants.is_directory_empty(path2)):
            store_Dataset.ds1_BDmaps(video_directory, video_files, path1, path2)
            x_train, x_test = activity_maps_files(path1, path2)
            print(f"Dataset {dataset_id}: Binarized/Decimated Activity Maps - Dataset shape:", np.shape(x_train), np.shape(x_test))
        
        else:
            x_train, x_test = activity_maps_files(path1, path2)
            print(f"Dataset {dataset_id}: Binarized/Decimated Activity Maps - Dataset shape:", np.shape(x_train), np.shape(x_test))

    else:

        if (constants.is_directory_empty(path1) and constants.is_directory_empty(path2)):
            store_Dataset.ds2_BDmaps(video_directory, video_files, path1, path2)
            x_train, x_test = activity_maps_files(path1, path2)
            print(f"Dataset {dataset_id}: Binarized/Decimated Activity Maps - Dataset shape:", np.shape(x_train), np.shape(x_test))
        
        else:
            x_train, x_test = activity_maps_files(path1, path2)
            print(f"Dataset {dataset_id}: Binarized/Decimated Activity Maps - Dataset shape:", np.shape(x_train), np.shape(x_test))

    return x_train, x_test