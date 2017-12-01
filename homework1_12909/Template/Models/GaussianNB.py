import numpy as np
import math

# make sure this class id compatable with sklearn's GaussianNB

class GaussianNB(object):
	
	def __init__(self ):
		# define all the model weights and state here
		self.classes = []	
		self.prior_prob = {}
		self.training_data = []
		self.training_target = []
		self.posterior_prob = []
		self.modalMapStats={}
		pass

	def fit(self,X,Y):
		self.training_data = X
		self.training_target = Y
		unique, counts = np.unique(Y, return_counts=True)
		self.classes = unique		
		#print len(unique)		
		self.prior_prob = {}	
		for i in range(0,len(unique)):	
			self.prior_prob[unique[i]]=(float(counts[i])/sum(counts))
			self.modalMapStats[unique[i]]=[]
		#self.print prior_prob
		pass


	
	def getvarandavg(self, feature_index,className):
		temp_list = []
		for i in range(0,len(self.training_data)):
			if(self.training_target[i] == className):			
				temp_list.append(self.training_data[i][feature_index])
		return np.var(temp_list), np.average(temp_list)			

	def calculate_prob(self, inputvector):
		likelihood = []
		for t in range(0,len(self.classes)):
			result = 0			
			for i in range(0,len(inputvector)):
				if(len(self.modalMapStats)!=0):
					if(len(self.modalMapStats[self.classes[t]])>=(i+1)):
						var= self.modalMapStats[self.classes[t]][i][0]
						avg = self.modalMapStats[self.classes[t]][i][1]
			                else:
						var,avg=self.getvarandavg(i,self.classes[t])
						temp = [var,avg]
						self.modalMapStats[self.classes[t]].append(temp)		
				

				
				if(var!=0.0):
					#print (1/(np.sqrt(2*np.pi*var))), (np.exp((-(inputvector[i]-avg)**2)/(2*var)))
					if((np.exp((-(inputvector[i]-avg)**2)/(2*var))) > 0.0 and (1/(math.sqrt(2*np.pi*var))) >0.0):
						#print np.log((1/(np.sqrt(2*np.pi*var)))* (np.exp((-(inputvector[i]-avg)**2)/(2*var))))
						try:						
							temp=math.log((1/(math.sqrt(2*np.pi*var)))* (math.exp((-(inputvector[i]-avg)**2)/(2*var))))	
							result = result + temp
						
						except ValueError:
							result =result
			if(result == '-inf'):
				result =0
			result = 1+result/10000			
			#print (1/(np.sqrt(2*np.pi*var))), (np.exp((-(inputvector[i]-avg)**2)/(2*var)))			
			likelihood.append(result)		
		return likelihood.index(max(likelihood))
			
	def predict(self, X):
		print len(self.classes)	
		summ =0
		count=0	
		for i in range(0,len(X)):
			k=self.calculate_prob(X[i])
			print i			
			#if(k==Y_R[2941+i]):
			#	summ=summ+1;
			#count=count+1			
			#print count,summ			
			self.posterior_prob.append(k)	
		pass
		return self.posterior_prob	 
		# return a numpy array of predictions
