import numpy as np #type:ignore
import os
import process_video
import constants

def for_training(row, array):
    return array[row][2] == 1

def ds1_opticalFlow(video_directory, video_files, training_path_opt, testing_path_opt, median_filter_size):

    for i in range(len(video_files)):

        if for_training(i, constants.array1):
            repetition = constants.repetitions1[i]
            gesture = constants.array1[i][0]
            type = constants.array1[i][1]

            magnitude = process_video.process_video(video_directory, video_files[i], "", median_filter_size)
            out = np.stack(magnitude, axis=0)
            #ux, uy = process_video.process_video(video_directory, video_files[i], "", median_filter_size)
            #out = np.stack((ux, uy), axis=0)
            os.chdir(training_path_opt)
            np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)
        
        else:
            repetition = constants.repetitions1[i]
            gesture = constants.array1[i][0]
            type = constants.array1[i][1]

            magnitude = process_video.process_video(video_directory, video_files[i], "", median_filter_size)
            out = np.stack(magnitude, axis=0)
            #ux, uy = process_video.process_video(video_directory, video_files[i], "", median_filter_size)
            #out = np.stack((ux, uy), axis=0)
            os.chdir(testing_path_opt)
            np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)


def ds2_opticalFlow(video_directory, video_files, training_path_opt, testing_path_opt, median_filter_size):
    
    for i in range(len(video_files)):

        if for_training(i, constants.array2):
            repetition = constants.repetitions2[i]
            gesture = constants.array2[i][0]
            type = constants.array2[i][1]
            
            if (repetition > 15):
                magnitude = process_video.process_video(video_directory, video_files[i], "left", median_filter_size)
                out = np.stack(magnitude, axis=0)
                #ux, uy = process_video.process_video(video_directory, video_files[i], "left", median_filter_size)
                #out = np.stack((ux, uy), axis=0)
                os.chdir(training_path_opt)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

            else:
                magnitude = process_video.process_video(video_directory, video_files[i], "right", median_filter_size)
                out = np.stack(magnitude, axis=0)
                #ux, uy = process_video.process_video(video_directory, video_files[i], "right", median_filter_size)
                #out = np.stack((ux, uy), axis=0)
                os.chdir(training_path_opt)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

        else:
            repetition = constants.repetitions2[i]
            gesture = constants.array2[i][0]
            type = constants.array2[i][1]
            
            magnitude = process_video.process_video(video_directory, video_files[i], "left", median_filter_size)
            out = np.stack(magnitude, axis=0)
            #ux, uy = process_video.process_video(video_directory, video_files[i], "left", median_filter_size)
            #out = np.stack((ux, uy), axis=0)
            os.chdir(testing_path_opt)
            np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)


def ds1_FSmaps(video_directory, video_files, path1, path2):
    
    for i in range(len(video_files)):

        if for_training(i, constants.array1):
            repetition = constants.repetitions1[i]
            gesture = constants.array1[i][0]
            type = constants.array1[i][1]

            activity_pattern = process_video.process_video2(video_directory, video_files[i], "")
            out = np.stack(activity_pattern, axis=0)
            os.chdir(path1)
            np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)    

        else:
            repetition = constants.repetitions1[i]
            gesture = constants.array1[i][0]
            type = constants.array1[i][1]

            activity_pattern = process_video.process_video2(video_directory, video_files[i], "")
            out = np.stack(activity_pattern, axis=0)
            os.chdir(path2)
            np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out) 


def ds2_FSmaps(video_directory, video_files, path1, path2):
   
    for i in range(len(video_files)):

        if for_training(i, constants.array2):
            repetition = constants.repetitions2[i]
            gesture = constants.array2[i][0]
            type = constants.array2[i][1]

            if (repetition > 15):
                activity_pattern = process_video.process_video2(video_directory, video_files[i], "left")
                out = np.stack(activity_pattern, axis=0)
                os.chdir(path1)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

            else:
                activity_pattern = process_video.process_video2(video_directory, video_files[i], "right")
                out = np.stack(activity_pattern, axis=0)
                os.chdir(path1)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

        else:
            repetition = constants.repetitions2[i]
            gesture = constants.array2[i][0]
            type = constants.array2[i][1]
            
            activity_pattern = process_video.process_video2(video_directory, video_files[i], "left")
            out = np.stack(activity_pattern, axis=0)
            os.chdir(path2)
            np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)


def ds1_BDmaps(video_directory, video_files, path1, path2):
    
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
            os.chdir(path1)
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
            os.chdir(path2)
            np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)



def ds2_BDmaps(video_directory, video_files, path1, path2):
    
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
                os.chdir(path1)
                np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)

            else:
                activity_pattern = process_video.process_video2(video_directory, video_files[i], "right")
                avg_ints_per_pattern = np.mean(activity_pattern)
                threshold = 2*avg_ints_per_pattern
                binarized_pattern = process_video.binarize_maps(activity_pattern, threshold)
                out = np.stack(binarized_pattern, axis=0)
                os.chdir(path1)
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
            os.chdir(path2)
            np.save('{}_{}_{}.npy'.format(int(gesture), int(type), int(repetition)), out)