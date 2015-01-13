#! /bin/python2

#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
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


# This retrieves all the directory names and puts them in a list
directory_names = list(set(glob.glob(os.path.join("train", "*"))\
 ).difference(set(glob.glob(os.path.join("train","*.*")))))

# This gets an random file
random_file = glob.glob(os.path.join(directory_names[5],"*.jpg"))[9]

df.append(af) will append af to the end, right of, df

# This bit of code will loop through all the files in a given path
import os
import glob

STANDARD_SIZE = (60, 60)
for folder in directory_names:
	# Something to note the name of the directory 
	current_label = folder.strip('train/')
	for plank_pic in glob.glob( os.path.join(folder, '*.jpg') ):
		print plank_pic
