# Sign Language Recognition

Recognize [American Sign Language (ASL)](https://en.wikipedia.org/wiki/American_Sign_Language) using Machine Learning.  
Currently, the following algorithms are supported:
 - [K-Nearest-Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
 - [Logistic Regression](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
 - [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)

The training images were retrieved from a video, filmed at `640x480` resolution using a mobile camera.

**Dependencies:**
 - [**OpenCV**](http://opencv.org/) 2.4.11, for image-processing.
 - [**Scikit-learn**](http://scikit-learn.org/) 0.18.1, for machine-learning algorithms.

**Usage:**
 1. Put all the training and testing images in a directory and update their paths in the config file *`common/config.py`*. The training images can be downloaded from [here](https://drive.google.com/folderview?id=0Bw239KLrN7zofmxvSmtsVHlrbkFRY1NwMjh2NFJGX1ZtY0lKOTR0REJnQnBUdVgyVDlMMkk&usp=sharing).  The test images are already present in the repo.
 2. (Optional) Generate the images in real-time from webcam - `capture_from_camera.py`.
 3. Generate image-vs-label mapping for all the training images - `generate_images_labels.py train`.
 4. Apply the image-transformation algorithms to the training images - `transform_images.py`.
 5. Train the model - `train_model.py <model-name>`. Note that the repo already includes pre-trained models for some algorithms serialized at *`data/generated/output/<model-name>/model-serialized-<model-name>.pkl`*.
 6. Generate image-vs-label mapping for all the test images - `generate_images_labels.py test`.
 7. Test the model - `predict_from_file.py <model-name>`.
 8. (Optional) Test the model on a live video stream from a webcam - `predict_from_camera.py`.

A sample workflow can be seen in *`run.sh`*.

**To-Do:**
 - Improve the command-line-arguments input mechanism.

