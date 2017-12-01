import os
import os.path
import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()



# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

X,Y1 = load_h5py(args.data)
Y=Y1
data_size = int(0.6*len(X));

def int_kernel(x1,x2):
	sigma=0.5
	f = np.exp((-1*np.square(np.linalg.norm(x1-x2)))/(2*np.square(sigma)))
	return f

#kernel must take as arguments 
#two matrices of shape (n_samples_1, n_features)
#(n_samples_2, n_features) and return a 
#kernel matrix of shape (n_samples_1, n_samples_2).
def gaussian_kernel(X1, X2, K=int_kernel):
    grammat = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            grammat[i,j] = K(x1,x2)
    return grammat

def predict(x,support_vectors_labels,support_vectors,coefs):
	y=[]
  	for i in x:
  	# 	result = 0
  	# 	for j,l,k in zip(coefs,support_vectors_labels,support_vectors):
 		# 	print j
 		# 	result+=j*int_kernel(j, i)
 		# #print result
 		# y.append(result)
 		dists = np.sum((support_vectors-i)**2,axis=1)
 		#dists = np.divide(dists,(2*np.square(0.5))
 		#print dists
 		probs = np.exp(-dists)
 		probs *= coefs
 		#probs /= np.sum(np.abs(probs))
 		hh =np.sum(probs)
 		
 	 	# if hh>0.95:
 	 	# 	hh = 1.1
		#print hh
 		
		y.append(hh)
	result=[]	
	for i in y:
		if i>=1:
			result.append(1)
		else:
			result.append(0)	
	return result     		
# def predict(x,theta,intercept):
# 	classification = np.sign(np.dot(x,theta)+intercept)
# 	print classification
# 	return classification
# # unique = np.unique(Y)
# # num = len(unique)
# # if n>2:
# # 	for i in range(0,num)
# # 		Y=Y1
# # 		for j in range(len(Y)):
# # 			if(Y[j]!=i):
# # 				Y[j]=1



unique = np.unique(Y)
num = len(unique)
print unique
if num>2:
	final_res = []
	for i in range(0,num):
		#print i
		Y = []
		for j in Y1:
			if(i==j):
				Y.append(1)
			else:	
				Y.append(0)
		clf= SVC(kernel =gaussian_kernel)		
		clt= clf.fit(X[0:data_size],Y[0:data_size])

		support_vectors=[]
		support_vectors_labels=[]
		for i in clf.support_:
			support_vectors.append(X[i])
			support_vectors_labels.append(Y[i])

		support_vectors = np.array(support_vectors)	
		coef = np.matmul(clf.dual_coef_,support_vectors)
		final_res.append(predict(X[data_size:len(X)],support_vectors_labels,support_vectors,np.reshape(clf.dual_coef_,[-1])))
		#print predict(X[data_size:len(X)],support_vectors_labels,support_vectors,np.reshape(clf.dual_coef_,[-1]))	
	
	pred = []
	for i in range(0,len(final_res[0])):
		if(final_res[0][i]==1):
			pred.append(0)
		elif(final_res[1][i]==1):
			pred.append(1)
		else:
			pred.append(2)
		
	k=0
	count=0
	print len(pred),len(Y1[data_size:len(Y1)])
	for i in Y1[data_size:len(Y1)]:
		if  i==pred[k]:
			count=count+1
		k=k+1	
	print count,len(Y1[data_size:len(Y1)])
	print 100*(count/float(len(Y1[data_size:len(Y1)])))
		




else:
	clf = SVC(kernel = gaussian_kernel)
	clt = clf.fit(X[0:data_size],Y1[0:data_size])
	# print clf.score(X[data_size:len(Y1)], Y1[data_size:len(Y1)]) 

	support_vectors=[]
	support_vectors_labels=[]
	for i in clf.support_:
		support_vectors.append(X[i])
		support_vectors_labels.append(Y[i])

	support_vectors = np.array(support_vectors)	
	coef = np.matmul(clf.dual_coef_,support_vectors)
	# print coef
	# temp = clf.score(X[data_size:len(X)],Y1[data_size:len(Y1)])
	# print temp
	temp = predict(X[data_size:len(X)],support_vectors_labels,support_vectors,np.reshape(clf.dual_coef_,[-1]))
	k=0
	count=0
	for i in Y1[data_size:len(Y1)]:
		if  i==temp[k]:
			count=count+1
		k=k+1	
	print count,len(Y1[data_size:len(Y1)])
	print 100*(count/float(len(Y1[data_size:len(Y1)])))


			
			
