from sklearn.metrics import confusion_matrix,roc_curve
from sklearn.externals import joblib
import numpy as np 
import os
import csv
from sklearn import svm
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import cPickle

def prec(num):
	return "%0.5f"%num

outfile = open("output/linear_svm_output.txt","a")

clf = svm.LinearSVC()
outfile.write(str(clf))

for dim in [10,20,30,40]:
	images=[]
	labels=[]
	name = str(dim)+"x"+str(dim)+".csv"
	with open("csv/"+name,'r') as file:
		reader = csv.reader(file,delimiter=',')
		for line in file:
			labels.append(line[0])
			line=line[2:] # Remove the label
			image=[int(pixel) for pixel in line.split(',')]
			images.append(np.array(image))
		
	print clf
	clf = svm.LinearSVC()
	
	kf = cross_validation.KFold(len(images),n_folds=10,indices=True, shuffle=True, random_state=4)
	outfile.write("Kfold on "+str(dim)+"x"+str(dim)+" dataset:\n\n")	
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
		predicted = clf.predict(testing_images)
		print prec(clf.score(testing_images, testing_labels))

#		outfile.write(prec(clf.score(testing_images, testing_labels))+'\n')
		outfile.write(roc_curve(testing_labels, predicted))
#		print confusion_matrix(testing_labels, predicted)
#		outfile.write(metrics.classification_report(testing_labels, predicted))

	print "\nDividing dataset using `train_test_split()` -:\n"
	outfile.write("\n\ntrain_test_split() on "+str(dim)+"x"+str(dim)+" dataset:\n\n")	
	training_images, testing_images, training_labels, testing_labels = cross_validation.train_test_split(images,labels, test_size=0.2, random_state=0)
	clf = clf.fit(training_images,training_labels)
	score = clf.score(testing_images,testing_labels)
	joblib.dump(clf, 'models/linearSVC_'+str(dim)+'x'+str(dim)+'.pkl')
	print "Dumped"	
	predicted = clf.predict(testing_images)
	print prec(score)
#	outfile.write(prec(clf.score(testing_images, testing_labels))+'\n')
	outfile.write(roc_curve(testing_labels, predicted))
#	print confusion_matrix(testing_labels, predicted)
#	outfile.write(metrics.classification_report(testing_labels, predicted))

