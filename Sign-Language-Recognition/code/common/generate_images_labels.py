#!/usr/bin/env python2
from os import walk
from os.path import join, splitext
from ntpath import basename


def get_images_labels_list(images_dir_path):
    """
    Recursively iterates through a directory and its subdirectories to list the info all the images found in it.
    Returns a list of dictionary where each dictionary contains `image_path` and `image_label`.
    """
    images_labels_list = []
    for (dirpath, dirnames, filenames) in walk(images_dir_path):
        for filename in filenames:
            image_path = join(dirpath, filename)
            image_label = splitext(basename(dirpath))[0]
            image_info = {}
            image_info['image_path'] = image_path
            image_info['image_label'] = image_label
            images_labels_list.append(image_info)
    return images_labels_list


def write_images_labels_to_file(images_labels_list, output_file_path):
    """
    Writes the list of images-labels to a file.
    """
    with open(output_file_path, "w") as output_file:
        for image_info in images_labels_list:
            image_path = image_info['image_path']
            image_label = image_info['image_label']
            line = image_path + "\t" + image_label + '\n'
            output_file.write(line)
