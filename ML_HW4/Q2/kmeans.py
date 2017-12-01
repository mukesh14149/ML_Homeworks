import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.manifold import TSNE

fname = 'seeds_dataset.txt'
X = []
tempY = []
#
# with open(fname) as f:
# 	for line in f:
# 		content = line.split(",")
# 		#print(len(content),content)
# 		if len(content) ==1:
# 			break
# 		temp = []
# 		for i in range(1,len(content)):
# 			temp.append(float(content[i]))

# 		X.append(temp)
# 		tempY.append(content[0])


with open(fname) as f:
	for line in f:
		content = line.split(",")
		#print(len(content),content)
		if len(content) ==1:
			break
		temp = []
		for i in range(0,len(content)-1):
			temp.append(float(content[i]))

		X.append(temp)
		tempY.append(content[len(content)-1])


X = np.array(X)		
print(X.shape)
mylist = list(set(tempY))
Y = []
for i in tempY:
	Y.append(mylist.index(i))
print(Y)

## Remove outlier using zscore

# temp = stats.zscore(X)
# X_train = []
# Y_train = []
# for hh,i in enumerate(temp):
# 	if all(i < 3 for i in my_list1) == True
# 		X_train.append(i)
		Y_train.append(Y[i])
# X = np.array(X_train)		
# Y = np.array(Y_train)

# #set k clustor
# k=3
# #Randomly pik k points from the dataset for centroid
# randscore = 0
# normal_mutual = 0
# adjust_mutual = 0

# for avg_run in range(0,50):
# 	print(avg_run)
# 	centroid = X[np.random.choice(len(X), size=k, replace=False)]
# 	#print(centroid)

# 	final_obj_res = []
# 	for ff in range(0,50):
# 		#empty list which holds samples in the corresponding centroid
# 		k_list = []
# 		for i in range(0,len(centroid)):
# 			k_list.append([])
# 		#find each sample distance from each centroid and put in the respective list	
				
# 		for i in X:
# 			mindis = np.linalg.norm(i-centroid[0])
# 			ind = 0
# 			for j in range(1,len(centroid)):
# 				#print("rrr",j)
# 				if(np.linalg.norm(i-centroid[j])<=mindis):
# 					ind = j
# 					mindis = np.linalg.norm(i-centroid[j])
# 			#print(ind)
# 			k_list[ind].append(i)		

# 		li = np.array(k_list)
# 		# print(np.array(li[0]).shape)
# 		# print(np.array(li[1]).shape)
# 		# print(np.array(li[2]).shape)
# 		# #update centroids based on mean
# 		for j,i in enumerate(li):
# 			c = np.mean(np.array(i),axis=0)
# 			centroid[j] = c

		
# 		obj_res = 0
# 		for ind, th in enumerate(k_list):
# 			for gh in th:
# 				obj_res += np.abs(np.linalg.norm(gh-centroid[ind]))
# 		final_obj_res.append(obj_res)


# 	#print(len(li))
# 	target_calculated = []
# 	for i in X:
# 		ind = 0
# 		flag=False
# 		for tt in range(0,len(li)):
# 			for j in li[tt]:
# 				if np.all(i==j):
# 					ind=(tt)
# 					flag=True
# 					break
# 			if flag==True:
# 				break

# 		target_calculated.append(ind)


# #PLot iteration vs cost
# # plt.plot(range(50),final_obj_res)
# # plt.xlabel("Iteration")
# # plt.ylabel("Cost")
# # plt.show()	


# 	#print(target_calculated)
# 	randscore +=adjusted_rand_score(Y, target_calculated)
# 	normal_mutual +=normalized_mutual_info_score(Y,target_calculated)
# 	adjust_mutual +=adjusted_mutual_info_score(Y,target_calculated)  		
# print(float(randscore)/5)
# print(float(normal_mutual)/5)
# print(float(adjust_mutual)/5)

# # plt.figure(1)
# # X_tsne = TSNE(learning_rate=100,verbose=1,n_components=2).fit_transform(X)
# # print X_tsne.shape
# # color=["#FF0000","#FFFF00","#808000","#000000", "#008000","#00FFFF","#0000FF","#800080","#FF00FF","#008080"]
# # plt.scatter(X_tsne[:,0], X_tsne[:,1],color=[color[i] for i in Y])

# # plt.figure(2)
# # color=["#FF0000","#FFFF00","#808000","#000000", "#008000","#00FFFF","#0000FF","#800080","#FF00FF","#008080"]
# # plt.scatter(X_tsne[:,0], X_tsne[:,1],color=[color[i] for i in target_calculated])
# # plt.show()	



# # #segmentation
# # k=7
# # 0.337095752045
# # 0.514785232556
# # 0.466425943921
# # k=2
# # 0.0815099514688
# # 0.283909099981
# # 0.135697992291
# # k=12
# # 0.348412236035
# # 0.576632482558
# # 0.491113404101


# ##Seeddataset
# # k=3
# # 0.714108597149
# # 0.700983266183
# # 0.696423091426
# # k=2
# # 0.467615297043
# # 0.552094017453
# # 0.428419507542

# # k=12
# # 0.265247190893
# # 0.520884852835
# # 0.337268455705

# ##Coloumn 
# # k=12
# # 0.194862501667
# # 0.407259232338
# # 0.260910046844
# # k=3
# # 0.311620347463
# # 0.420964329261
# # 0.412802244725

# # k=2
# # 0.29543068676
# # 0.422754796323
# # 0.332974727708


# # #iris
# # k=2
# # 0.539921829421
# # 0.679322701116
# # 0.519360805606

# # k=3
# # 0.665919394275
# # 0.720617426702
# # 0.700680096063

# # k=12
# # 0.334232641542
# # 0.626916076096
# # 0.409870320001
