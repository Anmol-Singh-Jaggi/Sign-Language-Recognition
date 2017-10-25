#!/usr/bin/env python2
import os
import sys
import csv

import numpy as np
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


from common.config import get_config


def print_with_precision(num):
    return "%0.5f" % num


def read_images_transformed(images_transformed_path):
    print("\nReading the transformed images file located at path '{}'...".format(
        images_transformed_path))

    images = []
    labels = []
    with open(images_transformed_path) as images_transformed_file:
        reader = csv.reader(images_transformed_file, delimiter=',')
        for line in reader:
            label = line[0]
            labels.append(label)
            image = line[1:]
            image_int = [int(pixel) for pixel in image]
            image = np.array(image_int)
            images.append(image)

    print("Done!\n")
    return images, labels


def generate_knn_classifier():
    num_neighbours = 10
    print("\nGenerating KNN model with number of neighbours = '{}'...".format(
        num_neighbours))
    classifier_model = KNeighborsClassifier(n_neighbors=num_neighbours)
    print("Done!\n")
    return classifier_model


def generate_logistic_classifier():
    print("\nGenerating Logistic-regression model...")
    classifier_model = linear_model.LogisticRegression()
    print("Done!\n")
    return classifier_model


def generate_svm_classifier():
    print("\nGenerating SVM model...")
    classifier_model = svm.LinearSVC()
    print("Done!\n")
    return classifier_model


def generate_classifier(model_name):
    classifier_generator_function_name = "generate_{}_classifier".format(
        model_name)
    return globals()[classifier_generator_function_name]()


def divide_data_train_test(images, labels, ratio):
    print("\nDividing dataset in the ratio '{}' using `train_test_split()`:".format(ratio))
    ret = train_test_split(images, labels, test_size=ratio, random_state=0)
    print("Done!\n")
    return ret


def main():
    model_name = sys.argv[1]
    if model_name not in ['svm', 'logistic', 'knn']:
        print("Invalid model-name '{}'!".format(model_name))
        return

    model_output_dir_path = get_config(
        'model_{}_output_dir_path'.format(model_name))
    model_stats_file_path = os.path.join(
        model_output_dir_path, "stats-{}.txt".format(model_name))
    print("Model stats will be written to the file at path '{}'.".format(
        model_stats_file_path))

    with open(model_stats_file_path, "w") as model_stats_file:
        images_transformed_path = get_config('images_transformed_path')
        images, labels = read_images_transformed(images_transformed_path)
        classifier_model = generate_classifier(model_name)

        model_stats_file.write("Model used = '{}'".format(model_name))
        model_stats_file.write(
            "Classifier model details:\n{}\n\n".format(classifier_model))
        training_images, testing_images, training_labels, testing_labels = divide_data_train_test(
            images, labels, 0.2)

        print("\nTraining the model...")
        classifier_model = classifier_model.fit(
            training_images, training_labels)
        print("Done!\n")

        model_serialized_path = get_config(
            'model_{}_serialized_path'.format(model_name))
        print("\nDumping the trained model to disk at path '{}'...".format(
            model_serialized_path))
        joblib.dump(classifier_model, model_serialized_path)
        print("Dumped\n")

        print("\nWriting model stats to file...")
        score = classifier_model.score(testing_images, testing_labels)
        model_stats_file.write(
            "Model score:\n{}\n\n".format(print_with_precision(score)))

        predicted = classifier_model.predict(testing_images)
        report = metrics.classification_report(testing_labels, predicted)
        model_stats_file.write(
            "Classification report:\n{}\n\n".format(report))
        print("Done!\n")

        print("\nFinished!\n")


if __name__ == '__main__':
    main()
