import numpy as np
import constants
import store_Dataset
import create_Dataset
import process_video
import os
import cv2


def for_training(row, array):
    return array[row][2] == 1

# Εδώ γίνεται το split του train και του test 80%-20%(104 numpy maps for training & 26 for testing)
def train_test_split(video_directory, video_files, training_path, testing_path, name):
    
    if name == 'full-resolution':
        for i in range(len(video_files)):

            if for_training(i, constants.array1):
                repetition = constants.repetitions1[i]
                gesture = constants.array1[i][0]
                type = constants.array1[i][1]
            
                activity_pattern = process_video.process_video2(video_directory, video_files[i], "")
                out = np.stack(activity_pattern, axis=0)
                os.chdir(training_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)    

            else:
                repetition = constants.repetitions1[i]
                gesture = constants.array1[i][0]
                type = constants.array1[i][1]

                activity_pattern = process_video.process_video2(video_directory, video_files[i], "")
                out = np.stack(activity_pattern, axis=0)
                os.chdir(testing_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)  

    else: 
        for i in range(len(video_files)):

            if for_training(i, constants.array1):
                repetition = constants.repetitions1[i]
                gesture = constants.array1[i][0]
                type = constants.array1[i][1]

                activity_pattern = process_video.process_video2(video_directory, video_files[i], "")
                avg_ints_per_pattern = np.mean(activity_pattern)
                threshold = 2*avg_ints_per_pattern
                binarized_pattern = process_video.binarize_maps(activity_pattern, threshold)
                out = np.stack(binarized_pattern, axis=0)
                os.chdir(training_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

            else:
                repetition = constants.repetitions1[i]
                gesture = constants.array1[i][0]
                type = constants.array1[i][1]

                activity_pattern = process_video.process_video2(video_directory, video_files[i], "")
                avg_ints_per_pattern = np.mean(activity_pattern)
                threshold = 2*avg_ints_per_pattern
                binarized_pattern = process_video.binarize_maps(activity_pattern, threshold)
                out = np.stack(binarized_pattern, axis=0)
                os.chdir(testing_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out) 


def train_test_split2(video_directory, video_files, training_path, testing_path, name):
    
    if name == 'full-resolution':
        for i in range(len(video_files)):

            if for_training(i, constants.array2):
                repetition = constants.repetitions2[i]
                gesture = constants.array2[i][0]
                type = constants.array2[i][1]

                if (repetition > 15):
                    activity_pattern = process_video.process_video2(video_directory, video_files[i], "left")
                    out = np.stack(activity_pattern, axis=0)
                    os.chdir(training_path)
                    np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

                else:
                    activity_pattern = process_video.process_video2(video_directory, video_files[i], "right")
                    out = np.stack(activity_pattern, axis=0)
                    os.chdir(training_path)
                    np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

            else:
                repetition = constants.repetitions2[i]
                gesture = constants.array2[i][0]
                type = constants.array2[i][1]
                
                activity_pattern = process_video.process_video2(video_directory, video_files[i], "left")
                out = np.stack(activity_pattern, axis=0)
                os.chdir(testing_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

    else:
        for i in range(len(video_files)):
 
            if for_training(i, constants.array2):
                repetition = constants.repetitions2[i]
                gesture = constants.array2[i][0]
                type = constants.array2[i][1]

                if (repetition > 15):
                    activity_pattern = process_video.process_video2(video_directory, video_files[i], "left")
                    avg_ints_per_pattern = np.mean(activity_pattern)
                    threshold = 2*avg_ints_per_pattern
                    binarized_pattern = process_video.binarize_maps(activity_pattern, threshold)
                    out = np.stack(binarized_pattern, axis=0)
                    os.chdir(training_path)
                    np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

                else:
                    activity_pattern = process_video.process_video2(video_directory, video_files[i], "right")
                    avg_ints_per_pattern = np.mean(activity_pattern)
                    threshold = 2*avg_ints_per_pattern
                    binarized_pattern = process_video.binarize_maps(activity_pattern, threshold)
                    out = np.stack(binarized_pattern, axis=0)
                    os.chdir(training_path)
                    np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)
        
            else:

                repetition = constants.repetitions2[i]
                gesture = constants.array2[i][0]
                type = constants.array2[i][1]
            
                activity_pattern = process_video.process_video2(video_directory, video_files[i], "left")
                avg_ints_per_pattern = np.mean(activity_pattern)
                threshold = 2*avg_ints_per_pattern
                binarized_pattern = process_video.binarize_maps(activity_pattern, threshold)
                out = np.stack(binarized_pattern, axis=0)
                os.chdir(testing_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)



def train_test_split_optflow(video_directory, video_files, training_path, testing_path, median_filter_size, dataset_id):
    if dataset_id == 1:
        
        for i in range(len(video_files)):

            if for_training(i, constants.array1):
                repetition = constants.repetitions1[i]
                gesture = constants.array1[i][0]
                type = constants.array1[i][1]
                
                ux, uy = process_video.process_video(video_directory, video_files[i], "", median_filter_size)
                out = np.stack((ux, uy), axis=0)
                os.chdir(training_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)    

            else:
                repetition = constants.repetitions1[i]
                gesture = constants.array1[i][0]
                type = constants.array1[i][1]

                ux, uy = process_video.process_video(video_directory, video_files[i], "", median_filter_size)
                out = np.stack((ux, uy), axis=0)
                os.chdir(testing_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)  

    else: 
        for i in range(len(video_files)):

            if for_training(i, constants.array1):
                repetition = constants.repetitions1[i]
                gesture = constants.array1[i][0]
                type = constants.array1[i][1]
                
                mag = process_video.process_video(video_directory, video_files[i], "", median_filter_size)
                out = np.stack(mag, axis=0)
                os.chdir(training_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)    

            else:
                repetition = constants.repetitions1[i]
                gesture = constants.array1[i][0]
                type = constants.array1[i][1]

                mag = process_video.process_video(video_directory, video_files[i], "", median_filter_size)
                out = np.stack(mag, axis=0)
                os.chdir(testing_path)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)  





# Φόρτωση των δεδομένων με βάση τα paths του train & test
def maps_files(train_path, test_path):

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

# έλεγχος για empty file και επιστροφή των x_train, x_test με τη βοήθεια της προηγούμενης συνάρτησης 
def load_maps(video_directory, video_files, training_path, testing_path, name, dataset_id):
    
    constants.create_directories1(training_path, testing_path)
   
    if name == 'full-resolution':
        if dataset_id == 1:
            if (constants.is_directory_empty(training_path) and constants.is_directory_empty(testing_path)):
                train_test_split(video_directory, video_files, training_path, testing_path, name)
                x_train, x_test = maps_files(training_path, testing_path)
                print(f"Dataset {dataset_id}: Activity Maps - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
                
            else:
                x_train, x_test = maps_files(training_path, testing_path)
                print(f"Dataset {dataset_id}: Activity Maps - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
        
        else:
            if (constants.is_directory_empty(training_path) and constants.is_directory_empty(testing_path)):
                train_test_split2(video_directory, video_files, training_path, testing_path, name)
                x_train, x_test = maps_files(training_path, testing_path)
                print(f"Dataset {dataset_id}: Activity Maps - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
                
            else:
                x_train, x_test = maps_files(training_path, testing_path)
                print(f"Dataset {dataset_id}: Activity Maps - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))

        return x_train, x_test

    else:
        if dataset_id == 1:
            if (constants.is_directory_empty(training_path) and constants.is_directory_empty(testing_path)):
                train_test_split(video_directory, video_files, training_path, testing_path, name)
                x_train, x_test = maps_files(training_path, testing_path)
                print(f"Dataset {dataset_id}: Decimated-Binarized Activity Maps - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
                
            else:
                x_train, x_test = maps_files(training_path, testing_path)
                print(f"Dataset {dataset_id}: Decimated-Binarized Activity Maps - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
        
        else:
            if (constants.is_directory_empty(training_path) and constants.is_directory_empty(testing_path)):
                train_test_split2(video_directory, video_files, training_path, testing_path, name)
                x_train, x_test = maps_files(training_path, testing_path)
                print(f"Dataset {dataset_id}: Decimated-Binarized Activity Maps - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
                
            else:
                x_train, x_test = maps_files(training_path, testing_path)
                print(f"Dataset {dataset_id}: Decimated-Binarized Activity Maps - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
    
        return x_train, x_test
    

def load_optflow(video_directory, video_files, training_path, testing_path, median_filter_size, dataset_id):
    
    constants.create_directories1(training_path, testing_path)
   
    if dataset_id == 1:
        if (constants.is_directory_empty(training_path) and constants.is_directory_empty(testing_path)):
            train_test_split_optflow(video_directory, video_files, training_path, testing_path, median_filter_size, dataset_id)
            x_train, x_test = maps_files(training_path, testing_path)
            print(f"Dataset {dataset_id}: Optical flow - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
                
        else:
            x_train, x_test = maps_files(training_path, testing_path)
            print(f"Dataset {dataset_id}: Optical flow - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
        
    else:
        if (constants.is_directory_empty(training_path) and constants.is_directory_empty(testing_path)):
            train_test_split_optflow(video_directory, video_files, training_path, testing_path, median_filter_size, dataset_id)
            x_train, x_test = maps_files(training_path, testing_path)
            print(f"Dataset {dataset_id}: Optical flow - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))
                
        else:
            x_train, x_test = maps_files(training_path, testing_path)
            print(f"Dataset {dataset_id}: Optical flow - Train shape, Test shape:", np.shape(x_train), np.shape(x_test))

    return x_train, x_test

    




# split του training file με τα 104 numpy files(8 maps per gesture) με ratio 75%-25%
# δηλαδή 6 maps για το train του μοντέλου και 2 για το validation
def train_val_split(sample_ratio, x_test, labels):

    #if name == "full-resolution":
    
    samples_per_gesture = [2] * 13
    num_gestures = len(samples_per_gesture)
    '''
        else:
            num_samples, height, width, num_channels = x_train.shape
            num_gestures = num_samples // 8

            x_train = x_train.reshape(num_gestures, 8, height, width, num_channels)
    '''

    test_maps = []
    test_labels = []
    validation_maps = []
    validation_labels = []

    start_index = 0

    for i in range(num_gestures):
        num_samples = samples_per_gesture[i]  # Should be 2 in this case
        gesture_samples = x_test[start_index:start_index + num_samples]
        gesture_labels = labels[start_index:start_index + num_samples]

        split_index = int(sample_ratio * num_samples)  # This will be 1 since num_samples = 2 and sample_ratio = 0.5

        test_maps.extend(gesture_samples[:split_index])
        validation_maps.extend(gesture_samples[split_index:])
        test_labels.extend(gesture_labels[:split_index])
        validation_labels.extend(gesture_labels[split_index:])

        start_index += num_samples

    test_maps = np.array(test_maps)
    test_labels = np.array(test_labels)
    validation_maps = np.array(validation_maps)
    validation_labels = np.array(validation_labels)
   


    return test_maps, test_labels, validation_maps, validation_labels


def train_val_split2(sample_ratio, x_test, labels):
 
    #samples_per_gesture = [20] + [24] * 9 + [23] + [24] * 2  # First gesture has 25, next 9 have 30, 10th has 29, last 2 have 29 for x_train
    samples_per_gesture = [5] + [6] * 12

    # Define the number of gestures.
    num_gestures = len(samples_per_gesture)
    test_maps = []
    test_labels = []
    validation_maps = []
    validation_labels = []

    start_index = 0

    for i in range(num_gestures):
        num_samples = samples_per_gesture[i]
        gesture_samples = x_test[start_index:start_index + num_samples]
        gesture_labels = labels[start_index:start_index + num_samples]

        split_index = int(sample_ratio * num_samples)

        test_maps.extend(gesture_samples[:split_index])
        validation_maps.extend(gesture_samples[split_index:])
        test_labels.extend(gesture_labels[:split_index])
        validation_labels.extend(gesture_labels[split_index:])

        start_index += num_samples

    test_maps = np.array(test_maps)
    test_labels = np.array(test_labels)
    validation_maps = np.array(validation_maps)
    validation_labels = np.array(validation_labels)

    return test_maps, test_labels, validation_maps, validation_labels


# Παρακάτω ξεκινά η επεξεργασία των βίντεο σχεδόν ίδια με την προηγούμενη μόνο
# αυτή τη φορά επιστρέφω activity maps με τα 3 κανάλια RGB και όχι σε grayscale μορφή 
def equal_frames(frame1,frame2):

    return np.array_equal(frame1,frame2)

def diff(frame1, frame2):

    return cv2.absdiff(frame1, frame2)

def process_channels(video_directory, video_file, channels):
    
    filtered_frames = []
    path = os.path.join(video_directory, video_file)

    cap = cv2.VideoCapture(path)

    width = int(cap.get(3))
    height = int(cap.get(4))

    crop_width = 42
    crop_height = 60

    previous_frame = None
    num_of_duplicates = 0
    num_of_frames = 0
   
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        
    while(cap.isOpened()):
  
        ret, frame = cap.read()

        if ret == True:

            current_frame = frame[crop_height:height-crop_height,crop_width:width-crop_width]

            if channels == 1:
                current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)

            if isinstance(previous_frame, np.ndarray):
                if equal_frames(current_frame, previous_frame):
                    num_of_duplicates += 1
                else:
                    num_of_frames += 1
                    if (num_of_frames == 16):
                       break
                    else:
                        previous_frame = current_frame
                        current_frame_filter = cv2.medianBlur(previous_frame, 3)
                        filtered_frames.append(current_frame_filter)
            else:
                previous_frame = current_frame
                current_frame_filter = cv2.medianBlur(previous_frame, 3)
                filtered_frames.append(current_frame_filter)
                num_of_frames += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
       

        #print(num_of_duplicates)
    filtered_frames = np.array(filtered_frames)
    #print(len(filtered_frames))
    #print(num_of_frames)
    if channels == 1:
        activity_map = np.zeros((height-2*crop_height, width-2*crop_width), dtype = np.uint8)

    else:
        activity_map = np.zeros((height-2*crop_height, width-2*crop_width, 3), dtype = np.uint8)

    for i in range(len(filtered_frames)-1):
        current_frame = filtered_frames[i]
        next_frame = filtered_frames[i+1]
        sub = diff(next_frame, current_frame)
        activity_map += sub

    cv2.normalize(activity_map, activity_map, 0, 255, cv2.NORM_MINMAX)

    return activity_map

#x_train, x_test = train_test_split(constants.video_directory1, constants.sorted_video_files1, constants.ds1_train_path, constants.ds1_test_path, 'full-resolution')
#x_train, x_test = load_maps(constants.video_directory1, constants.sorted_video_files1, constants.ds1_train_path, constants.ds1_test_path, "full-resolution")