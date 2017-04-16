#!/usr/bin/env python2
import os


def get_config(key):
    global config
    return config[key]


def add_config(key, value):
    global config
    config[key] = value


def merge_paths(key, path):
    return os.path.join(get_config(key), path)


def populate_config():
    add_config('data_root_path', "../data")

    add_config('training_images_dir_path', merge_paths(
        'data_root_path', 'images/train'))
    add_config('testing_images_dir_path', merge_paths(
        'data_root_path', 'images/test'))

    add_config('generated_data_dir_path', merge_paths(
        'data_root_path', 'generated'))

    add_config('training_images_labels_path', merge_paths(
        'generated_data_dir_path', 'training_images_labels.txt'))
    add_config('testing_images_labels_path', merge_paths(
        'generated_data_dir_path', 'testing_images_labels.txt'))

    add_config('images_transformed_path', merge_paths(
        'generated_data_dir_path', 'images_transformed.csv'))

    add_config('output_dir_path', merge_paths(
        'generated_data_dir_path', 'output'))

    for model_name in ["knn", "logistic", "svm"]:
        key = 'model_{}_output_dir_path'.format(model_name)
        value = merge_paths('output_dir_path', '{}'.format(model_name))
        add_config(key, value)
        key = 'model_{}_serialized_path'.format(model_name)
        value = merge_paths('model_{}_output_dir_path'.format(
            model_name), 'model-serialized-{}.pkl'.format(model_name))
        add_config(key, value)


config = {}
populate_config()
