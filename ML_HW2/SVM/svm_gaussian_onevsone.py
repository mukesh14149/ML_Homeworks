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
	final_res = {}
	for i in range(0,num):
		for j in range(0,num):
			if i!=j:
				Y = []
				X_temp = []
				jj=0
				for a in Y1[0:data_size]:
					print i,j
					if(i==a):
						Y.append(1)
						X_temp.append(X[jj])
						
					elif(j==a):	
						Y.append(0)
						X_temp.append(X[jj])
					
					jj = jj+1
				print Y
				clf= SVC(kernel =gaussian_kernel)
				clt= clf.fit(X_temp[0:data_size],Y[0:data_size])
				support_vectors=[]
				support_vectors_labels=[]
				for hh in clf.support_:
					support_vectors.append(X[hh])
					support_vectors_labels.append(Y[hh])

				support_vectors = np.array(support_vectors)	
				coef = np.matmul(clf.dual_coef_,support_vectors)
				print coef
				final_res[(i,j)]=predict(X[data_size:len(X)],support_vectors_labels,support_vectors,np.reshape(clf.dual_coef_,[-1]))
			
	print final_res
	pred = []
	for s in range(0,len(X[data_size:len(X)])):
		for i in range(0,num):
			flag=0
			for j in range(i+1,num):
				if(final_res[i,j][s] ==0):
					flag=1
			if flag==0:
				pred.append(i)

	k=0
	count=0
	for i in Y1[data_size:len(Y1)]:
		if  i==pred[k]:
			count=count+1
		k=k+1	
	print count,len(Y1[data_size:len(Y1)])
	print 100*(count/float(len(Y1[data_size:len(Y1)])))
		
