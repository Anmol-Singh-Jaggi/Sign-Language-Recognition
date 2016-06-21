import numpy as np 
import os
import csv
from sklearn import svm
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import cPickle

def prec(num):
	return "%0.5f"%num

images=[]
labels=[]
with open("20x20_temp.csv",'r') as file:
	reader = csv.reader(file,delimiter=',')
	for line in file:
		labels.append(line[0])
		line=line[2:] # Remove the label
		image=[int(pixel) for pixel in line.split(',')]
		images.append(np.array(image))
		
clf = linear_model.LogisticRegression()

print clf
kf = cross_validation.KFold(len(images),n_folds=10,indices=True, shuffle=True, random_state=4)
print "\nDividing dataset using `Kfold()` -:\n\nThe training dataset has been divided into " + str(len(kf)) + " parts\n"
for train, test in kf:
	training_images=[]
	training_labels=[]
	for i in train:
		training_images.append(images[i])
		training_labels.append(labels[i])
	testing_images=[]
	testing_labels=[]
	for i in test:
		testing_images.append(images[i])
		testing_labels.append(labels[i])
	clf.fit(training_images,training_labels)
	print prec(clf.score(testing_images, testing_labels))

print "\nDividing dataset using `train_test_split()` -:\n"
training_images, testing_images, training_labels, testing_labels = cross_validation.train_test_split(images,labels, test_size=0.2, random_state=0)
clf = clf.fit(training_images,training_labels)
score = clf.score(testing_images,testing_labels)
print prec(score)
