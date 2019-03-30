import os
import sys
import traceback
import logging

import cv2
from sklearn.externals import joblib

from common.config import get_config
from common.image_transformation import apply_image_transformation
from common.image_transformation import resize_image


logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)


def get_image_from_label(label):
    testing_images_dir_path = get_config('testing_images_dir_path')
    image_path = os.path.join(testing_images_dir_path, label, '001.jpg')
    image = cv2.imread(image_path)
    return image


def main():
    model_name = sys.argv[1]
    if model_name not in ['svm', 'logistic', 'knn']:
        logger.error("Invalid model-name '{}'!".format(model_name))
        return
    logger.info("Using model '{}'...".format(model_name))

    model_serialized_path = get_config(
        "model_{}_serialized_path".format(model_name))
    logger.info("Model deserialized from path '{}'".format(
                model_serialized_path))

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to capture image!")
            continue
        frame = resize_image(frame, 400)
        cv2.imshow("Webcam recording", frame)
        try:
            frame = apply_image_transformation(frame)
            frame_flattened = frame.flatten()
            classifier_model = joblib.load(model_serialized_path)
            predicted_labels = classifier_model.predict(frame_flattened)
            predicted_label = predicted_labels[0]
            logger.info("Predicted label = {}".format(predicted_label))
            predicted_image = get_image_from_label(predicted_label)
            predicted_image = resize_image(predicted_image, 200)
            cv2.imshow("Prediction = '{}'".format(
                predicted_label), predicted_image)
        except Exception:
            exception_traceback = traceback.format_exc()
            logger.error("Error applying image transformation")
            logger.debug(exception_traceback)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    logger.info("The program completed successfully !!")


if __name__ == '__main__':
    main()
