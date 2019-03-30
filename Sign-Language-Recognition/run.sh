# Activate virtual environment
#pipenv shell
cd code

logs_dir_path="./logs"
mkdir -p "${logs_dir_path}"

# Modify this as per your need.
export LOGLEVEL=INFO
export SHOW_PROGRESS_BAR=True

printf "%s\n" "----- Generating training image labels... -----"
script_name="generate_images_labels"
log_path="${logs_dir_path}/${script_name}.log"
python "${script_name}.py" train 2>&1 | tee "${log_path}"
printf "\n%s\n\n" "----- Done! -----"

printf "%s\n" "----- Transforming images... -----"
# It is normal to get errors on some of the images.
# Can turn off the progress bar inside
script_name="transform_images"
log_path="${logs_dir_path}/${script_name}.log"
python "${script_name}.py" 2>&1 | tee "${log_path}"
printf "\n%s\n\n" "----- Done! -----"

# Can be "svm"/"knn"/"logistic"
model_name="logistic"

printf "%s\n" "----- Building model... -----"
script_name="train_model"
log_path="${logs_dir_path}/${script_name}.log"
python "${script_name}.py" "${model_name}" train 2>&1 | tee "${log_path}"
printf "\n%s\n\n" "----- Done! -----"

printf "%s\n" "----- Generating testing image labels... -----"
script_name="generate_images_labels"
log_path="${logs_dir_path}/${script_name}.log"
python "${script_name}.py" test train 2>&1 | tee "${log_path}"
printf "\n%s\n\n" "----- Done! -----"

printf "%s\n" "----- Predicting from file... -----"
# It is normal to get errors on some of the images.
script_name="predict_from_file"
log_path="${logs_dir_path}/${script_name}.log"
python "${script_name}.py" "${model_name}" train 2>&1 | tee "${log_path}"
printf "\n%s\n\n" "----- Done! -----"
