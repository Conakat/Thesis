# Part 2: Hand Gesture Recognition using A-mode Ultrasound Signals

This branch contains the A-mode ultrasound gesture recognition experiments.

A Prototypical Network is trained using episodic few-shot learning on A-mode ultrasound signals.
Support samples are selected using angle-based decimation, ensuring uniform coverage of motion
phases within each gesture class, while query samples are drawn from unseen data to avoid
information leakage.

The dataset consists of 6 hand gestures, with approximately 500 samples per gesture (3000 samples in total).
For each gesture, 400 samples are used for training, while the remaining 100 are split into validation
(first 25) and testing (remaining 75). Each sample has dimensions (8, 960).

Three evaluation protocols are considered:
* Same subject – new repetitions
* Same subject – new gestures
* New subject – new gestures

Each experiment constructs episodic tasks by sampling gesture classes and forming support/query
sets accordingly.

Datasets are not publicly available due to privacy constraints.

