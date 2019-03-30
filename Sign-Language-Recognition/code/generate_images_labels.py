"""
Contains code to generate the <image-path> vs <image-label> list for a set of
images and write it to disk.
"""
import sys
import logging
import os

from common.config import get_config


logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)


def get_images_labels_list(images_dir_path):
    """
    Recursively iterates through a directory and its subdirectories to list
    the info all the images found in it.
    Returns a list of dictionary where each dictionary contains `image_path`
    and `image_label`.
    """
    images_labels_list = []
    logger.info('Images directory - "{}"'.format(images_dir_path))
    for (dirpath, dirnames, filenames) in os.walk(images_dir_path):
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            image_label = os.path.splitext(os.path.basename(dirpath))[0]
            image_info = {}
            image_info['image_path'] = image_path
            image_info['image_label'] = image_label
            images_labels_list.append(image_info)
    return images_labels_list


def write_images_labels_to_file(images_labels_list, output_file_path):
    """
    Writes the list of images-labels to a file.
    """
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        for image_info in images_labels_list:
            image_path = image_info['image_path']
            image_label = image_info['image_label']
            line = image_path + "\t" + image_label + '\n'
            output_file.write(line)


def main():
    images_source = sys.argv[1]
    if images_source not in ['train', 'test']:
        logger.error("Invalid image-source '{}'!".format(images_source))
        return
    images_dir_path = get_config('{}ing_images_dir_path'.format(images_source))
    images_labels_path = get_config(
        '{}ing_images_labels_path'.format(images_source))

    logger.info("Gathering info about images at path '{}'..."
                .format(images_dir_path))
    images_labels_list = get_images_labels_list(images_dir_path)
    logger.info("Done!")

    logger.info("Writing images labels info to file at path '{}'...".format(
        images_labels_path))
    write_images_labels_to_file(images_labels_list, images_labels_path)
    logger.info("Done!")


if __name__ == '__main__':
    main()
