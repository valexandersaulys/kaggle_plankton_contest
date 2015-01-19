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
import glob
import os
from sklearn import cross_validation
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
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.externals import joblib # For saving

os.system("taskset -p 0xff %d" % os.getpid())

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

print "rescale the images to be 25x25, build the bits for storing info"
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
num_features = imageSize + 1 # for our ratio

# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
x_train = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y_train = np.zeros((num_rows))

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
			files.append(nameFileImage)
			image = resize(image, (maxPixel, maxPixel))
			# Store the rescaled image pixels and the axis ratio
			x_train[i, 0:imageSize] = np.reshape(image, (1, imageSize))        
			# Store the classlabel
			y_train[i] = label
			i += 1		# increment the entry number
			# report progress for each 5% done  
			report = [int((j+1)*num_rows/20.) for j in range(20)]
			if i in report: print np.ceil(i *100.0 / num_rows), "% done"
	label += 1

#
# # # # # #
#

# So now we have X, our features vectors with its corresponding label vector y
# Think of the MNIST contest for further calculations	

# Then we build a classifier based on the information
print "Building Classifier"
classifier= RandomForestClassifier(n_estimators=500, 
								criterion='gini', 
								max_depth=None, 
								min_samples_split=2,
								min_samples_leaf=1, 
								max_features='auto', 
								max_leaf_nodes=None, 
								bootstrap=True, 
								oob_score=False,
								n_jobs=4, 
								random_state=None, 
								verbose=0, 
								min_density=None, 
								compute_importances=None)

# Then we build the predictors
model_rf = classifier.fit(x_train,y_train)
print 'Done!'

print "Now writing to disk with Joblib"
os.mkdir('model_rf_500n')
joblib.dump(model_rf, 'model_rf_500n/model_rf_500n.pk') 

# Now we look at the test data
numOfTest = 0	# To initialize the variable
for pic in os.listdir('test'):
	numOfTest += 1

# We'll build out array for testing
num_rows = numOfTest # one row for each image in the training dataset
num_features = imageSize + 1 # for our ratio
x_test = np.zeros((num_rows, num_features), dtype=float) 
y_test = np.zeros((num_rows))

test_files = [] 	# allocating the space for the test files to be outputted


print "Reading Test Information => 130400 pics"
i = 0
for pic in os.listdir('test'):	
	print "pic #%d" % (i+1)
	# I'm assuming all the files are jpegs        
	# Read in the images and create the features
	nameFileImage = "{0}{1}{2}".format('test', os.sep, pic)            
	image = imread(nameFileImage, as_grey=True)
	test_files.append(nameFileImage[5:])		# We need to take out the "test/" part
	image = resize(image, (maxPixel, maxPixel))
	# Store the rescaled image pixels and the axis ratio
	x_test[i, 0:imageSize] = np.reshape(image, (1, imageSize))
	i += 1		# Increment the entry number
   

# Then we output everything into a neat and tidy csv 
"""
for the output:
	we need the image_name.jpg + predicted probabilities
"""
print 'Building Predictions'
probabilities = model_rf.predict_proba(x_test)

print 'Getting Ready to Submit'
headers = []
headers.append('image')
for i in range(len(dClasses)):
	headers.append(dClasses.values()[i]) 


size_of_submit = np.zeros((len(probabilities[:,0]), len(probabilities[0,:]) + 1))
submit = pd.DataFrame(size_of_submit, columns=headers)

#print len(probabilities[0,:])		# 121 number of columns
#print len(probabilities[:,0])		# 130400 number of rows

print "Writing to DataFrame, this will take the longest"
for i in range(num_rows):
	print "Writing Pic#%d" % ( i )
	submit.iloc[i,0] = test_files[i]
	submit.iloc[i,1:] = probabilities[i,:]

print "Writing Results to csv..."
submit.to_csv("submittion_attempt_one.csv", sep=',', index=False)



print "--- Done in %s seconds! ---" % (time.time() - start_time)



