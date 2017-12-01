import numpy as np


# make sure this class id compatable with sklearn's LogisticRegression

class LogisticRegression(object):

	def __init__(self, penalty='l2' , C=1.0 , max_iter=100 , verbose=0):
		# define all the model weights and state here
		self.training_data = []
		self.training_target = []
		self.max_iter = max_iter
		self.weight =-1
		self.output =[]
		self.C = C
		pass


	def sigmoid(self, x):
 		return 1 / (1 + np.exp(-x))

	def fit(self,X , Y):
		self.training_data = X
		self.training_target = Y
        	
   		W = np.zeros(X.shape[1])	
		print len(W)    	
		for i in range(self.max_iter): 
        		preds = self.sigmoid(np.dot(X, W))
			error =  Y - preds       		
			gradient = np.dot(X.T, error)
			W += self.C * gradient
			self.weight=W        				
		pass

	def predict(self,X):
		self.output=self.sigmoid(np.dot(X,self.weight))
		type(self.output)		
		return self.output	  
		# return a numpy array of preds
