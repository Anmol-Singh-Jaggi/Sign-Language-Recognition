"""
Takes a set of images as inputs, transforms them using multiple algorithms to
make it suitable for ingestion into ML routines, then finally outputs them
to disk.
"""
import csv
import traceback
import logging
import os

import numpy as np
import cv2
from tqdm import tqdm

from common.config import get_config
from common.image_transformation import apply_image_transformation


logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)

show_progress_bar = os.environ.get("SHOW_PROGRESS_BAR", "True") == "True"


def write_frame_to_file(frame, frame_label, writer):
    """
    Convert the multi-dimensonal array of the image to a one-dimensional one
    and write it to a file, along with its label.
    """
    logger.debug("Writing frame to file...")
    flattened_frame = frame.flatten()
    output_line = [frame_label] + np.array(flattened_frame).tolist()
    writer.writerow(output_line)
    logger.debug("Done!")


def main():
    images_transformed_path = get_config('images_transformed_path')
    os.makedirs(os.path.dirname(images_transformed_path), exist_ok=True)
    with open(images_transformed_path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        training_images_labels_path = get_config('training_images_labels_path')
        with open(training_images_labels_path, 'r') as file:
            lines = file.readlines()
        if show_progress_bar:
            progress_bar = tqdm(total=len(lines), desc='Transforming images',
                                unit=' pics')
        for line in lines:
            logger.debug("\n\n" + line.strip())
            image_path, image_label = line.split()
            # Read the input image.
            frame = cv2.imread(image_path)
            # `frame` is a HxW numpy ndarray of triplets (pixels),
            # where H and W are the dimensions of the input image.
            # cv2.imshow("Original", frame)
            try:
                if show_progress_bar:
                    progress_bar.update()
                frame = apply_image_transformation(frame)
                write_frame_to_file(frame, image_label, writer)
            except Exception:
                # Its normal to get errors on some images!!
                exception_traceback = traceback.format_exc()
                logger.error("Error applying image transformation to image "
                             "'{}'".format(image_path))
                logger.debug(exception_traceback)
                continue
            # cv2.waitKey(1000)
    cv2.destroyAllWindows()
    if show_progress_bar:
        progress_bar.close()
    logger.info("The program completed successfully !!")


if __name__ == '__main__':
    main()
