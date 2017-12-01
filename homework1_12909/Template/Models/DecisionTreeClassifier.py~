import numpy as np


# make sure this class id compatable with sklearn's DecisionTreeClassifier

class DecisionTreeClassifier(object):

	def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
		# define all the model weights and state here
		self.classes = []		
		self.prior_prob = {}		
		self.H_Y = 0
		self_bucket_size = 10		
		pass				
	
	def compute_H_Y(self):
		for i in range(0,len(self.prior_prob)):
			self.H_Y = self.H_Y - (self.prior_prob[i]*np.log2(self.prior_prob[i]))
		print self.H_Y
		pass

	#def compute_H_Y_given_X(self,Y,W,X):
			
	
	def fit(self,X , Y):
		print np.ndarray.max(X)		
		self.training_data = X
		self.training_target = Y
		unique, counts = np.unique(Y, return_counts=True)
		self.classes = unique		
		print len(unique)		
		self.prior_prob = {}	
		for i in range(0,len(unique)):	
			self.prior_prob[unique[i]]=(float(counts[i])/sum(counts))		
		self.compute_H_Y()
				
		pass

	def predict(X ):
		pass 
		# return a numpy array of predictions
