#!/usr/bin/env bash

cd code

# Activates the Anaconda virtual environment on my system.
# You would probably want to comment this line.
source env/bin/activate ./env

logs_dir_path="./logs"
mkdir -p "${logs_dir_path}"

model_name="svm"

printf "%s\n" "----- Predicting from file... -----"
./predict_from_file.py "${model_name}" > "${logs_dir_path}/predict_from_file.log" 2>&1
printf "\n%s\n\n" "----- Done! -----"
