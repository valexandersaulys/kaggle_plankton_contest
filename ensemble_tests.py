#! /bin/python2
# Specifically I'm using Python 2.7.6

# # # # # # # # # # # # 
# # # # # # TO DO:
#	-> Going to use this kaggle code as a base
#	-> Try Neural Networks, SVM, RandomForest (tutorial on Kaggle for that), Kmeans
#	-> How to combine different methods?
#	-> predict_proba returns an array of shape = [n_samples, n_classes]
#
#	-> Idea for ensemble learning:
#		-> Could look at which one is most likely given the circumstances
#			and then output that one as the final.
#
#	-> Tuning needs to be done for output
#		-> Create a dictionary for outputting
#


# libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
import glob
import os
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
import warnings
import time
import cv2

# My Own Python Library Files
import ensemble_rank
import useful_selfmade_library as usl

warnings.filterwarnings("ignore")

from sklearn.externals import joblib # For saving

os.system("taskset -p 0xff %d" % os.getpid())
start_time = time.time()

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("train", "*"))\
 ).difference(set(glob.glob(os.path.join("train","*.*")))))

dClasses = {}		# A dictionary would be far more convenient
names_of_classes = []

print "get the total training images"
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

print "rescale the images to be 50x50, build the bits for storing info"
maxPixel = 50
imageSize = (maxPixel * maxPixel) 
num_rows = numberofImages # one row for each image in the training dataset
num_features = imageSize + 1 # for our ratio

# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
# First position is our label
training = np.zeros((num_rows, num_features+1), dtype=float)

files = []		# to hold all the files
# Generate training data
i = 0    
label = 1	# start the label at 1

# List of string of class names
namesClasses = {}		# A dictionary would be far more convenient

print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
	# Append the string class name for each class
	currentClass = folder[6:]
	dClasses[label] = currentClass

	for fileNameDir in os.walk(folder):   
		for fileName in fileNameDir[2]:
			# I'm assuming all the files are jpegs        
			# Read in the images and create the features
			nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
			image = imread(nameFileImage, as_grey=True)
			
			# Resize image and threshold to binary (strictly B&W)
			image = resize(image, (maxPixel, maxPixel))
			thresh = threshold_otsu(image)
			binary = image > thresh
			files.append(nameFileImage)
			
			# Store the rescaled image pixels and the axis ratio
			training[i,1:2501] = np.reshape(binary, (1, imageSize))
			# Store the classlabel
			training[i,0] = label
			i += 1		# increment the entry number
			# report progress for each 5% done  
			report = [int((j+1)*num_rows/20.) for j in range(20)]
			if i in report: print np.ceil(i *100.0 / num_rows), "% done"
	label += 1

train_data,valid_data = usl.DataPartition(training,0.7)
x_train = train_data[:,1:imageSize]
y_train = train_data[:,0]
x_valid = valid_data[:,1:imageSize]
y_valid = valid_data[:,0]

#
# # # # # # # # # # # # # # # #
#

# So now we have X, our features vectors with its corresponding label vector y

# Then we build a classifier based on the information
super_models = []
print "Test Classifiers"
print "First Random Forest"

param_dist = {"max_depth": [3, None],
              "max_features": [range(1, 11)[0], "auto", "log2", "sqrt"],
              "min_samples_split": range(1, 11),
              "min_samples_leaf": range(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = GridSearchCV(RandomForestClassifier(), 
								param_grid=param_dist,
								cv=3, n_jobs=6).fit(x_train,y_train)



print "Best Random Forest Model"
print random_search.grid_scores_
print random_search.best_score_
print random_search.best_estimator_
super_models.append(random_search.best_estimator_)
print ""

print "Next SVM"

param_dist = {"C": [1.0, 0.1,0.01,0.001],
			"kernel": ['rbf'],
			'degree': [2,3,4,5],
			'gamma' : [0.0,0.1,0.01,0.001],
			'shrinking': [True,False],
			'probability': [False,True],
			'tol':[0.001,0.0001,0.00001]}

# run randomized search
n_iter_search = 20
random_search = GridSearchCV(SVC(), 
							param_grid=param_dist,
							cv=3, n_jobs=6).fit(x_train,y_train)

print "Best SVM Model"
print random_search.grid_scores_
print random_search.best_score_
print random_search.best_estimator_
super_models.append(random_search.best_estimator_)
print ""

print "Next Adaboost"

param_dist = {"n_estimators": [50,100,250,500],
			"learning_rate": [1.0,2.0,3.0,4.0,5.0],
			'algorithm': ['SAMME.R','SAMME']}

sklearn.ensemble.AdaBoostClassifier(base_estimator=None, 
							n_estimators=50, learning_rate=1.0, 
							algorithm='SAMME.R', random_state=None)

# run randomized search
n_iter_search = 20
random_search = GridSearchCV(AdaBoostClassifier(), 
								param_grid=param_dist,
								cv=3, n_jobs=6).fit(x_train,y_train)

print "Best Adaboost Model"
print random_search.grid_scores_
print random_search.best_score_
print random_search.best_estimator_
super_models.append(random_search.best_estimator_)
print ""

jeff = ensemble_rank.Ensemble_Kay(x_train=x_train,
						y_train=y_train,
						x_valid=x_valid,
						y_valid=y_valid,
						models=super_models)

msf,bsf = jeff.return_best_models()
print msf
print bsf

y_preds = jeff.calculate_new_data(x_test)
