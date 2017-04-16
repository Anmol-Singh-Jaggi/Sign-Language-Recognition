#!/usr/bin/env python2
import sys
import traceback

import cv2
from sklearn.externals import joblib

from common.config import get_config
from common.image_transformation import apply_image_transformation


def main():
    model_name = sys.argv[1]
    if model_name not in ['svm', 'logistic', 'knn']:
        print("Invalid model-name '{}'!".format(model_name))
        return

    print("Using model '{}'...".format(model_name))

    model_serialized_path = get_config(
        'model_{}_serialized_path'.format(model_name))
    print("Model deserialized from path '{}'".format(model_serialized_path))

    testing_images_labels_path = get_config('testing_images_labels_path')
    with open(testing_images_labels_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print("\n\n" + line.strip())
            image_path, image_label = line.split()
            frame = cv2.imread(image_path)
            try:
                frame = apply_image_transformation(frame)
                frame_flattened = frame.flatten()
                classifier_model = joblib.load(model_serialized_path)
                predicted_labels = classifier_model.predict(frame_flattened)
                predicted_label = predicted_labels[0]
                print("Predicted label = {}".format(predicted_label))

                if image_label != predicted_label:
                    print("Incorrect prediction!!")
                    cv2.waitKey(5000)
            except Exception:
                exception_traceback = traceback.format_exc()
                print("Error while applying image transformation on image path '{}' with the following exception trace:\n{}".format(
                    image_path, exception_traceback))
                continue
    cv2.destroyAllWindows()
    print "The program completed successfully !!"


if __name__ == '__main__':
    main()
