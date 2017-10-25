# Sign Language Recognition

Recognize [American Sign Language (ASL)](https://en.wikipedia.org/wiki/American_Sign_Language) using Machine Learning.  
Currently, the following algorithms are supported:
 - [K-Nearest-Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
 - [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
 - [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)

The training images were retrieved from a video, filmed at `640x480` resolution using a mobile camera.

**Dependencies:**
 - [**OpenCV**](http://opencv.org/) 2.4.11, for image-processing.
 - [**Scikit-learn**](http://scikit-learn.org/) 0.18.1, for machine-learning algorithms.

**Usage:**

For training a dataset of your own, do the following steps:  
 1. Put all the training and testing images in a directory and update their paths in the config file *`code/common/config.py`*. Optionally, you can generate the images in real-time from webcam - `capture_from_camera.py`.
 2. Generate image-vs-label mapping fsor all the training images - `generate_images_labels.py train`.
 3. Apply the image-transformation algorithms to the training images - `transform_images.py`.
 4. Train the model - `train_model.py <model-name>`.
 6. Generate image-vs-label mapping for all the test images - `generate_images_labels.py test`.
 7. Test the model - `predict_from_file.py <model-name>`. Optionally, you can test the model on a live video stream from a webcam - `predict_from_camera.py`.

A sample workflow can be seen in *`run.sh`*.

However, if you wish not to use your own dataset, you can skip some of these steps and use the pre-trained models trained using [this dataset](https://drive.google.com/drive/folders/0Bw239KLrN7zoNkU5elZMRkc4TU0?usp=sharing):

 1. Download and replace the contents of the directory `data/generated` from [here](https://drive.google.com/drive/folders/0Bw239KLrN7zoelVsMVU5SnEwa0k?usp=sharing). It contains the serialized model files, the transformed images as well as the image-vs-label mapping files.
 2. Test the model - `predict_from_file.py <model-name>`.

A sample workflow can be seen in *`run_quick.sh`*.

**To-Do:**
 - Improve the command-line-arguments input mechanism.
 - Add progress bar while transforming images.
 - Add logger.
