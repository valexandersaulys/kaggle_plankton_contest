import numpy as np

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix

from sklearn.cross_validation import cross_val_score

# # # # # # # # # # # # # # # #
# # # #	Assumpations:
# # # #	-> That Python can pass a list of models and call them
# # # # 
# # # # # # # # # # # # 



class  Nomaste():	
	
	def __init__(self, x_train,y_train,x_valid,y_valid,models=[]):
		self.x_train = x_train
		self.y_train = y_train
		self.x_valid = x_valid
		self.y_valid = y_valid
		self.models = models	# List of model parameters, think sklearn
		self.__calculate_all_predictions()
		self.__calculate_all_scores()
		
	def change_models(self, models=[]):
		# This takes all the models in the form of parameters from SciKit
		self.models = models
		
	def get_prediction(self, p):
		return self.models[p]
		
	def combine_models(self, lmods=[]):
		#
		#	Someway to combine models with probabilities
		#	Maybe not necesary?
		#
		pass	
	
	# Private Methods
		
	def __calculate_all_predictions(self):
		self.predictions = []	# List of arrays
		for model in self.models:
			self.predictions.append(model.predict_proba(self.x_train))
	
	def __calculate_all_scores(self):
		self.pred_scores = []	# List of values
		for model in self.models:
			self.pred_scores.append(cross_val_score(
												estimator=model(),
												X=x_valid,
												y=y_valid,
												cv=5))

	def __add_predictions(self,a1,a2=[]):
		running_sum = a1
		for i in range(len(a2)):
			running_sum += a2
		return(running_sum)


			
			
class Ensemble_Jay(Nomaste):
	####
	###
	#		Calculates the best model if given many 
	#		I don't know if this actually works
	###
	####

	def __ensemble_rank():
		for i in range(len(self.models)):
			if i==0:
				best_so_far = self.pred_scores[i]
				model_so_far = [self.models(i)]
			for j in range(i):
				for g in range(j):
					if self.__performance(i,range(g)) > best_so_far:
						model_so_far = [i]
						model_so_far.append(range(g))
		return(model_so_far,best_so_far)

	def __performance(self):	
		if others==[0]:
			ginger = np.mean(cross_validation(passed_model).fit(x_valid,y_valid))	#Something like this anyway
			asm = Accuracy_Score(ginger)
			return asm
		else:
			ginger = [passed_num]
			jerry = []
			for i in range(len(others)):
				ginger.append(i)
			for spice in ginger:
				jerry.append(mean(CrossValidate(spice).fit(x_valid,y_valid)))


class Ensemble_Kay(Nomaste):
	####
	###
	#		Calculates the best ensemble model combining all given
	###
	####

	def calculate_new_data(self, x_test):
		y_preds = self.__calculate_ensemble(x_test)
		return(y_preds)

	def calc_those_probabilities(self,x_test):
		y_probs = self.__calc_proba(x_test)
		return(y_probs)

	def __calculate_ensemble(self):
		preds = []
		for model in self.models:
			preds.append(model.predict(self.x_valid,self.y_valid))
		return_preds = self.__add_predictions(a1=preds[0],a2=preds[1:])
		return(return_preds)

	def __calculate_ensemble(self,x_test):
		preds = []
		for model in self.models:
			preds.append(model.predict(x_test))
		return_preds = self.__add_predictions(a1=preds[0],a2=preds[1:])
		return(return_preds)

	def __calc_proba(self,x_test):
		probs = []	# List of arrays
		for model in self.models:
			probs.append(model.predict_proba(x_test))
		return_probs = self.__add_predictions(a1=probs[0],a2=probs[1:])
		return(return_probs)

	def confusion_matrix(self):
		y_preds = self.__calculate_ensemble()
		confusion_matrix(self.y_valid,self.y_preds)

	def confusion_matrix(self, x_v, y_v):
		y_p = self.__calculate_ensemble(x_test=self.x_v)
		confusion_matrix(y_v,y_p)

