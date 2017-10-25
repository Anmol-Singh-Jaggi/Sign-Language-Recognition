#!/usr/bin/env bash

cd code

# Activates the Anaconda virtual environment on my system.
# You would probably want to comment this line.
source env/bin/activate ./env

logs_dir_path="./logs"
mkdir -p "${logs_dir_path}"

printf "%s\n" "----- Generating training image labels... -----"
./generate_images_labels.py train
printf "\n%s\n\n" "----- Done! -----"

printf "%s\n" "----- Transforming images... -----"
./transform_images.py > "${logs_dir_path}/transform_images.log" 2>&1
printf "\n%s\n\n" "----- Done! -----"

model_name="svm"

printf "%s\n" "----- Building model... -----"
./train_model.py "${model_name}"
printf "\n%s\n\n" "----- Done! -----"

printf "%s\n" "----- Generating testing image labels... -----"
./generate_images_labels.py test
printf "\n%s\n\n" "----- Done! -----"

printf "%s\n" "----- Predicting from file... -----"
./predict_from_file.py "${model_name}" > "${logs_dir_path}/predict_from_file.log" 2>&1
printf "\n%s\n\n" "----- Done! -----"
