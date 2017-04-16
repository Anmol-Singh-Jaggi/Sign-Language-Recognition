#!/usr/bin/env python2
"""
Takes a set of images as inputs, transforms them using multiple algorithms to
make it suitable for ingestion into ML routines, then finally outputs them
to disk.
"""
import csv
import traceback

import numpy as np
import cv2

from common.config import get_config
from common.image_transformation import apply_image_transformation


def write_frame_to_file(frame, frame_label, writer):
    """
    Convert the multi-dimensonal array of the image to a one-dimensional one
    and write it to a file, along with its label.
    """
    print("Writing frame to file...")

    flattened_frame = frame.flatten()
    output_line = [frame_label] + np.array(flattened_frame).tolist()
    writer.writerow(output_line)

    print("Done!")


def main():
    images_transformed_path = get_config('images_transformed_path')
    with open(images_transformed_path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')

        training_images_labels_path = get_config('training_images_labels_path')
        with open(training_images_labels_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            print("\n\n" + line.strip())
            image_path, image_label = line.split()

            # Read the input image.
            frame = cv2.imread(image_path)
            # `frame` is a HxW numpy ndarray of triplets (pixels), where H and W are
            # the dimensions of the input image.
            # cv2.imshow("Original", frame)
            try:
                frame = apply_image_transformation(frame)
                write_frame_to_file(frame, image_label, writer)
            except Exception:
                exception_traceback = traceback.format_exc()
                print("Error while applying image transformation on image path '{}' with the following exception trace:\n{}".format(
                    image_path, exception_traceback))
                continue
            # cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print "The program completed successfully !!"


if __name__ == '__main__':
    main()
