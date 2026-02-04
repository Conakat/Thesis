import create_Dataset
import algorithms
import constants
import matplotlib.pyplot as plt #type:ignore

dataset_id = [1, 2]
median_filter_size = [3, 7, 9, 12, 15]

labels_ds1 = create_Dataset.labels_DS1()
labels_ds2 = create_Dataset.labels_DS2()

y_train_ds1, y_test_ds1 = create_Dataset.labels_SVM(labels_ds1)
y_train_ds2, y_test_ds2 = create_Dataset.labels_SVM(labels_ds2)

x_fullres_train, x_fullres_test = create_Dataset.load_activityMaps(constants.video_directory1, constants.sorted_video_files1, constants.path_for_kNN1_FS_train, constants.path_for_kNN1_FS_test, dataset_id[0])
#algorithms.kNN_classification(x_fullres_train, x_fullres_test, y_train_ds2, y_test_ds2)

x_bd_train, x_bd_test = create_Dataset.load_BDmaps(constants.video_directory1, constants.sorted_video_files1, constants.path_for_kNN1_BD_train, constants.path_for_kNN1_BD_test, dataset_id[0])
#algorithms.kNN_classification(x_bd_train, x_bd_test, y_train_ds2, y_test_ds2)

#print(x_bd_train.shape, x_bd_test.shape)
#x_train, x_test = create_Dataset.load_opticalFlow(constants.video_directory2, constants.sorted_video_files2, constants.training_path_opt2_15_med, constants.testing_path_opt2_15_med, median_filter_size[4], dataset_id[1])
#algorithms.svm_implementation(x_train, y_train_ds2, x_test, y_test_ds2)


num_classes = 13
classes = ["all finger r", "all finger f", "all finger e", "thumb f", "thumb e", "index f", "index e", "middle f", "middle e", "ring f", "ring e", "pingy f", "pingy e"]

for i in range(num_classes):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Display full-resolution activity map, viridis
    axs[0].imshow(x_fullres_train[i*8], cmap='gray')  # Adjust the colormap as needed
    axs[0].set_title(f'{classes[i]} - Full Resolution')

        # Display binarized/decimated activity map
    axs[1].imshow(x_bd_train[i*8], cmap='gray')  # Assuming binarized, adjust the colormap as needed
    axs[1].set_title(f'{classes[i]} - Binarized/Decimated')

        # Adjust layout and show the plots for the current class
    plt.tight_layout()
    plt.show()

#x_train_mag1, x_test_mag1 = create_Dataset.load_opticalFlow(constants.video_directory1, constants.sorted_video_files1, constants.training_path_mag1_15, constants.testing_path_mag1_15, median_filter_size[4], dataset_id[0])
#algorithms.svm_implementation(x_train_mag1, y_train_ds1, x_test_mag1, y_test_ds1)
