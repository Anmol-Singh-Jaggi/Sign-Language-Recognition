#!/usr/bin/env python2
"""
Contains code to generate the <image-path> vs <image-label> list for a set of images and write it to disk.
"""
import sys

from common.config import get_config

from common.generate_images_labels import get_images_labels_list
from common.generate_images_labels import write_images_labels_to_file


def main():
    images_source = sys.argv[1]
    if images_source not in ['train', 'test']:
        print("Invalid image-source '{}'!".format(images_source))
        return

    images_dir_path = get_config('{}ing_images_dir_path'.format(images_source))
    images_labels_path = get_config(
        '{}ing_images_labels_path'.format(images_source))

    print("Gathering info about images at path '{}'...".format(images_dir_path))
    images_labels_list = get_images_labels_list(images_dir_path)
    print("Done!")

    print("Writing images labels info to file at path '{}'...".format(
        images_labels_path))
    write_images_labels_to_file(images_labels_list, images_labels_path)
    print("Done!")


if __name__ == '__main__':
    main()
