import numpy as np #type: ignore
import os

video_directory1 = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\DS1"
video_files1 = [f for f in os.listdir(video_directory1) if f.endswith(".mp4")]

video_directory2 =r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\DS2"
video_files2 = [f for f in os.listdir(video_directory2) if f.endswith(".mp4")]

# TRAINING-TEST PATHS/DIFFERENT MEDIAN FILTER DS1
training_path_opt1_15_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Training optical flow DS1 15"
testing_path_opt1_15_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Testing optical flow DS1 15"

training_path_opt1_3_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Training optical flow DS1 3"
testing_path_opt1_3_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Testing optical flow DS1 3"

training_path_opt1_15_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Training optical flow DS1 15"
testing_path_opt1_15_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Testing optical flow DS1 15"

training_path_opt1_3_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Training optical flow DS1 3"
testing_path_opt1_3_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Testing optical flow DS1 3"

training_path_mag1_15_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Training magnitude DS1 15"
testing_path_mag1_15_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Testing magnitude DS1 15"

training_path_mag1_3_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Training magnitude DS1 3"
testing_path_mag1_3_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Testing magnitude DS1 3"

training_path_mag1_15_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Training magnitude DS1 15"
testing_path_mag1_15_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Testing magnitude DS1 15"

training_path_mag1_3_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Training magnitude DS1 3"
testing_path_mag1_3_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Testing magnitude DS1 3"

path_for_kNN1_FS_train = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Full Resolution Maps DS1-Train"
path_for_kNN1_FS_test = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Full Resolution Maps DS1-Test"
path_for_kNN1_BD_train = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Binarized-Decimated Maps DS1 Train"
path_for_kNN1_BD_test = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Binarized-Decimated Maps DS1 Test"

# paths for CNN 

ds1_train_path_fs = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Training maps for CNN"
ds1_test_path_fs = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Testing maps for CNN"

ds1_train_path_comp = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Optflow comp train for CNN"
ds1_test_path_comp = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Optflow comp test for CNN"

ds1_train_path_mag = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Optflow mag train for CNN"
ds1_test_path_mag = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Optflow mag test for CNN"

ds1_train_path_bd = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Training DBmaps for CNN"
ds1_test_path_bd = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Testing DBmaps for CNN"

ds2_train_path_fs = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Training maps 2 for CNN"
ds2_test_path_fs = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Testing maps 2 for CNN"

ds2_train_path_comp = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Optflow comp train 2 for CNN"
ds2_test_path_comp = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Optflow comp test 2 for CNN"

ds2_train_path_mag = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Optflow mag train 2 for CNN"
ds2_test_path_mag = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Optflow mag test 2 for CNN"

# TRAINING-TEST PATHS/DIFFERENT MEDIAN FILTER DS2
training_path_opt2_15_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Training optical flow DS2 15"
testing_path_opt2_15_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Testing optical flow DS2 15"

training_path_opt2_3_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Training optical flow DS2 3"
testing_path_opt2_3_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Testing optical flow DS2 3"

training_path_opt2_15_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Training optical flow DS2 15"
testing_path_opt2_15_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Testing optical flow DS2 15"

training_path_opt2_3_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Training optical flow DS2 3"
testing_path_opt2_3_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Testing optical flow DS2 3"

training_path_mag2_15_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Training magnitude DS2 15"
testing_path_mag2_15_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Testing magnitude DS2 15"

training_path_mag2_3_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Training magnitude DS2 3"
testing_path_mag2_3_med = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Testing magnitude DS2 3"

#for checking the initial repetitions i used
training_path_mag2_3_med_in = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-initial Training magnitude DS2 3"
testing_path_mag2_3_med_in = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Initial Testing magnitude DS2 3"

training_path_opt2_3_med_in = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-initial Training optical flow DS2 3"
testing_path_opt2_3_med_in = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Medianblur-Initial Testing optical flow DS2 3"

training_path_mag2_15_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Training magnitude DS2 15"
testing_path_mag2_15_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Testing magnitude DS2 15"

training_path_mag2_3_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Training magnitude DS2 3"
testing_path_mag2_3_gb = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Gaussianblur-Testing magnitude DS2 3"

path_for_kNN2_FS_train = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Full Resolution Maps DS2-Train"
path_for_kNN2_FS_test = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Full Resolution Maps DS2-Test"
path_for_kNN2_BD_train = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Binarized-Decimated Maps DS2 Train"
path_for_kNN2_BD_test = r"C:\Users\ΚΩΝΣΤΑΝΤΙΝΟΣ\AppData\Local\Programs\Python\Python310\Binarized-Decimated Maps DS2 Test"


def create_directories1(training_path, testing_path):
    if not os.path.exists(training_path):
        os.makedirs(training_path, exist_ok=True)
    
    if not os.path.exists(testing_path):
        os.makedirs(testing_path, exist_ok=True)

def create_directories2(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def is_directory_empty(directory):
    return not os.listdir(directory)

def custom_sort(file_name):
    # Split the filename by underscore
    parts = file_name.split('_')
    # Return a tuple (x, y, z) with z as an integer
    return int(parts[0]), int(parts[1]), int(parts[2][:-4])  # Convert x, y, and z to integers

sorted_video_files1 = sorted(video_files1, key=custom_sort)

sorted_video_files2 = sorted(video_files2, key=custom_sort)

array_FS_ds1 = np.zeros((len(video_files1), 3))

for i in range(len(video_files1)):
    array_FS_ds1[i][2] = True

array_FS_ds2 = np.zeros((len(video_files2), 3))

for i in range(len(video_files2)):
    array_FS_ds2[i][2] = True

indeces1 = []
for video in sorted_video_files1:
    category, type = video.split(".")
    indeces1.append(category)

indeces1 = np.array(indeces1)

repetitions1 = []
array1 = np.zeros((len(video_files1), 3))

for i in range(len(indeces1)):
    array1[i] = indeces1[i].split("_")
    repetitions1.append(int(array1[i][2]))

for i in range(len(array1)):
    #if array1[i][2] == 10:
    # 2 & 7 (& 4) samples kala results
    if (array1[i][2] == 4 or array1[i][2] == 9): #or array1[i][2] == 8):
        array1[i][2] = False
    else:
        array1[i][2] = True

indeces2 = []
for video in sorted_video_files2:
    category, type = video.split(".")
    indeces2.append(category)

indeces = np.array(indeces2)
array2 = np.zeros((len(video_files2), 3))
repetitions2 = []

for i in range(len(indeces2)):
    array2[i] = indeces2[i].split("_")
    repetitions2.append(int(array2[i][2]))

#print(repetitions2)

first_index = 25
second_index = len(array2)
#print(second_index)

# 17, 19, 21, 23, 25
for i in range(first_index):
    if (array2[i][2]==17 or array2[i][2]==19 or array2[i][2]==21 or array2[i][2]==23 or array2[i][2]==25):
        array2[i][2] = False
    else:
        array2[i][2] = True

 # 22, 24, 25, 27, 28, 30
for i in range(first_index, second_index):
    if (array2[i][2]==22 or array2[i][2]==24 or array2[i][2]==25 or array2[i][2]==27 or array2[i][2]==28 or array2[i][2]==30):
        array2[i][2] = False 
    else:
        array2[i][2] = True