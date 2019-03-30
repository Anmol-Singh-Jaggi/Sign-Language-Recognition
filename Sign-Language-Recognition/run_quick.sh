# Activate virtual environment
#pipenv shell
cd code

logs_dir_path="./logs"
mkdir -p "${logs_dir_path}"

# Modify this as per your need.
export LOGLEVEL=INFO

# Can be "svm"/"knn"/"logistic"
model_name="svm"

printf "%s\n" "----- Predicting from file... -----"
python predict_from_file.py "${model_name}" 2>&1 | tee "${logs_dir_path}/predict_from_file.log" 
printf "\n%s\n\n" "----- Done! -----"
