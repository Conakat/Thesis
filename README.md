# Part 1: Hand Gesture Recognition using B-mode Ultrasound Images

Dataset: 
* B-mode ultrasound videos collected during experimentation of 13 finger gestures
* The finger gestures contain extension, flexion of each finger and alltogether, and a rest state
* Two datasets were generated:
  - Dataset 1: 130 samples obtained from one hand 
  - Dataset 2: 384 samples obtained from both hands

The videos were processed with the use of OpenCV functions and numpy library led to the following data represantations:
* Activity maps (deformation maps)
* Binarized/Decimated activity maps
* Horizontal and Vertical elements of Optical Flow
* Magnitude of Optical Flow

Processing pipeline:
* Preprocessing of ultrasound videos
* Extraction of the data representations
* Train / Validation / Test split
* Model training and hyperparameter tuning
* Evaluation using classification accuracy and confusion matrices

Models used:
* k-NN on activity maps
* SVM with optical features
* CNNs trained and evaluated with all forms of data

Results:
* CNNs achieved the highest performance, while the classical ML models showed competitive results when combined with handcrafted features. Especially the SVM model achieved higher accuracy than the CNNs when the input was the magnitude, specifically in Dataset 2.
