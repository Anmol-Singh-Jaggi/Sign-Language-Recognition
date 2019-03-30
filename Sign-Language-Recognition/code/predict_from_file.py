import sys
import traceback
import logging
import os

import cv2
from sklearn.externals import joblib

from common.config import get_config
from common.image_transformation import apply_image_transformation


logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)


def main():
    model_name = sys.argv[1]
    if model_name not in ['svm', 'logistic', 'knn']:
        logger.error("Invalid model-name '{}'!".format(model_name))
        return

    logger.info("Using model '{}'...".format(model_name))

    model_serialized_path = get_config(
        'model_{}_serialized_path'.format(model_name))
    logger.info("Model deserialized from path '{}'".
                format(model_serialized_path))

    testing_images_labels_path = get_config('testing_images_labels_path')
    with open(testing_images_labels_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line:
                continue
            image_path, image_label = line.split()
            frame = cv2.imread(image_path)
            try:
                frame = apply_image_transformation(frame)
                frame_flattened = frame.flatten()
                classifier_model = joblib.load(model_serialized_path)
                predicted_labels = classifier_model.predict([frame_flattened])
                predicted_label = predicted_labels[0]
                logger.info('"{}" {} ---> {}'.format(image_path, image_label,
                            predicted_label))
                if image_label != predicted_label:
                    log_msg = "Incorrect prediction '{}' instead of '{}'\n)"
                    logger.error(log_msg.format(predicted_label, image_label))
                    cv2.waitKey(5000)
            except Exception:
                exception_traceback = traceback.format_exc()
                logger.error("Error applying image transformation to image "
                             "'{}'".format(image_path))
                logger.debug(exception_traceback)
                continue
    cv2.destroyAllWindows()
    logger.info("The program completed successfully !!")


if __name__ == '__main__':
    main()
