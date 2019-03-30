"""
Contains code to capture images from a live webcam recording.
"""

import cv2
import os
import logging

from common.config import get_config


logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)


def capture_images(camera, num_images, num_seconds_to_wait, output_dir_path):
    """
    Captures images from a live webcam feed, and stores them to disk.
    :param camera: The VideoCapture object.
    :param num_images: The number of images to capture.
    :param num_seconds_to_wait: The number of seconds to wait between 2
        consecutive captures.
    :param output_dir_path: The path of the directory in which to save
        all the captured images.
    :returns: None
    """
    for num_image in range(1, num_images + 1):
        logger.info("\n\nTaking image #{}...".format(num_image))
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to capture image!")
            continue

        logger.info("Displaying the image taken...")
        cv2.imshow("win", frame)
        logger.info("Done!")

        if num_seconds_to_wait != -1:
            logger.info("Waiting for {} seconds...".
                        format(num_seconds_to_wait))
            cv2.waitKey(num_seconds_to_wait * 1000)
            logger.info("Done!")

        if output_dir_path:
            logger.info("Writing the captured image to file...")
            output_file_name = "{}.png".format(num_image)
            output_file_path = os.path.join(output_dir_path, output_file_name)
            ret = cv2.imwrite(output_file_path, frame)
            if not ret:
                logger.error("Error writing the image to the file path '{}'".
                             format(output_file_path))
            logger.info("Done!")


def main():
    camera = cv2.VideoCapture(0)

    num_images = 5
    num_seconds_to_wait = 1
    generated_data_dir_path = get_config('generated_data_dir_path')
    camera_captures_path = os.path.join(
        generated_data_dir_path, "camera_captures")
    capture_images(camera, num_images, num_seconds_to_wait,
                   camera_captures_path)

    logger.info("\n\nReleasing the camera...")
    camera.release()
    cv2.destroyAllWindows()
    logger.info("The program completed successfully !!")


if __name__ == '__main__':
    main()
