# Sign Language Recognition

Recognize [American Sign Language (ASL)](https://en.wikipedia.org/wiki/American_Sign_Language) using Machine Learning.  
Currently, the following algorithms are supported:
 - [K-Nearest-Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
 - [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
 - [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)

The [training images](https://drive.google.com/drive/folders/0Bw239KLrN7zoNkU5elZMRkc4TU0?usp=sharing) were retrieved from a video, filmed at `640x480` resolution using a smartphone camera.

**Setup:**
 - Install **`Python3`** (last tested on Python3.7).
 - Install [pipenv](https://pipenv.readthedocs.io/en/latest/).
 - In the project root directory, execute `pipenv sync`.

**Usage:**

You can directly start classifying new images using the pre-trained models (the `.pkl` files in `data/generated/output/<model_name>/`) trained using [this dataset](https://drive.google.com/drive/folders/0Bw239KLrN7zoNkU5elZMRkc4TU0?usp=sharing):

      python predict_from_file.py <model-name>

Note that the pre-generated model files do not contain the file for `knn` due to its large size.  
If you want to use `knn`, then download it separately from [here](https://drive.google.com/open?id=0Bw239KLrN7zoMWRCRjBTUUhtY1U) and place it in `data/generated/output/knn/`.  
The models available by default are `svm` and `logistic`.

The above workflow can be executed using *`run_quick.sh`*.

---------------------------------------------------------

However, if you wish to use your own dataset, do the following steps:  
 1. Put all the training and testing images in a directory and update their paths in the config file *`code/common/config.py`*.  
    (Or skip to use the default paths which should also work).  
    Optionally, you can generate the images in real-time from webcam - `python capture_from_camera.py`.
 2. Generate image-vs-label mappings for all the training images - `python generate_images_labels.py train`.
 3. Apply the image-transformation algorithms to the training images - `python transform_images.py`.
 4. Train the model - `python train_model.py <model-name>`. Model names can be `svm`/`knn`/`logistic`.
 6. Generate image-vs-label mapping for all the test images - `python generate_images_labels.py test`.
 7. Test the model - `python predict_from_file.py <model-name>`.  
    Optionally, you can test the model on a live video stream from a webcam - `python predict_from_camera.py`.  
    (If recording, then make sure to have the same background and hand alignment as in the training images.)

All the python commands above have to be executed from the `code/` directory.  
The above workflow can be executed using *`run.sh`*.

**To-Do:**
 - Improve the command-line-arguments input mechanism.
 - ~~Add progress bar while transforming images.~~
 - ~~Add logger.~~