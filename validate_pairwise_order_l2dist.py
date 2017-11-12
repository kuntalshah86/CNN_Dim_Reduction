#This code is for validation of pair-wise distance oedering within the class
import numpy
import random
import pickle

#Python functions to measure distance between different samples using different distance metrics
def __euclidian_dist(vec1, vec2):
    assert(vec1.shape == vec2.shape), 'Vectors are of different size. Cannot perform euclidean distance'
    return numpy.sqrt(numpy.sum(numpy.square(vec1-vec2)))

def __l1_dist(vec1, vec2):
    assert(vec1.shape == vec2.shape), 'Vectors are of different size. Cannot perform absolute distance'
    return numpy.sum(numpy.abs(vec1-vec2))
    
def __max_dist(vec1, vec2):
    assert(vec1.shape == vec2.shape), 'Vectors are of different size. Cannot perform max distance'
    return numpy.max(numpy.abs(vec1-vec2))
    
#def __dot_prod(vec1, vec2):
#    assert(vec1.shape == vec2.shape), 'Vectors are of different size. Cannot perform dot product'
#    y11 = np.sqrt(np.sum(np.power(vec1,2)))
#    y12 = np.sqrt(np.sum(np.power(vec2,2)))
#    if np.isnan(np.sum(vec1*vec2/y11/y12)):
#       return 0
#    else:
#       return np.sum(vec1*vec2/y11/y12)

def __Jaccard(vec1, vec2):
    assert(vec1.shape == vec2.shape), 'Vectors are of different size. Cannot perform Jaccard index'
    y11 = numpy.sum((numpy.min([1+vec1, 1+vec2],0)))   # to make entries of vectors positive
    y12 = numpy.sum((numpy.max([1+vec1, 1+vec2],0)))   # to make entries of vectors positive
    return 1-y11/y12
	
classes=['paisley','plain','floral','vertical','horizontal','diagonal','spotted','gingham'];
label_dict=dict(zip(classes,range(len(classes))));
label_dict_reverse=dict(zip(range(len(classes)),classes));

X=numpy.load('feature_matrix.dat');
X_transformed_256=numpy.load('X_transformed_feature_dimension_256.dat');
X_transformed_512=numpy.load('X_transformed_feature_dimension_512.dat');
X_transformed_1024=numpy.load('X_transformed_feature_dimension_1024.dat');
X_transformed_2048=numpy.load('X_transformed_feature_dimension_2048.dat');
X_transformed_4096=numpy.load('X_transformed_feature_dimension_4096.dat');
y=numpy.load('class_matrix.dat');

for k in sorted(label_dict.values()):

	indices = [index for index in range(len(y)) if y.item(index) == k];
	
	f1=open("pairwise_eucliddist_sorted_orig_dim_%s.dat" % label_dict_reverse[k],"wb");
	f2=open("pairwise_eucliddist_sorted_PCA256_%s.dat" % label_dict_reverse[k],"wb");
	f3=open("pairwise_eucliddist_sorted_PCA512_%s.dat" % label_dict_reverse[k],"wb");
	f4=open("pairwise_eucliddist_sorted_PCA1024_%s.dat" % label_dict_reverse[k],"wb");
	f5=open("pairwise_eucliddist_sorted_PCA2048_%s.dat" % label_dict_reverse[k],"wb");
	f6=open("pairwise_eucliddist_sorted_PCA4096_%s.dat" % label_dict_reverse[k],"wb");
	
	count=0;
	pair=[];
	dist_orig=[];
	dist_PCA256=[];
	dist_PCA512=[];
	dist_PCA1024=[];
	dist_PCA2048=[];
	dist_PCA4096=[];
	
	
	for i in range(0,len(indices)):
		for j in range(i+1,len(indices)):
			#Calculate pair wise distances between (i,j) sample pair within class
			#Using tuples
			pair.append(numpy.array([i,j]));
			dist_orig.append(__euclidian_dist(X[indices[i]],X[indices[j]]));
			dist_PCA256.append(__euclidian_dist(X_transformed_256[indices[i]],X_transformed_256[indices[j]]));
			dist_PCA512.append(__euclidian_dist(X_transformed_512[indices[i]],X_transformed_512[indices[j]]));
			dist_PCA1024.append(__euclidian_dist(X_transformed_1024[indices[i]],X_transformed_1024[indices[j]]));
			dist_PCA2048.append(__euclidian_dist(X_transformed_2048[indices[i]],X_transformed_2048[indices[j]]));
			dist_PCA4096.append(__euclidian_dist(X_transformed_4096[indices[i]],X_transformed_4096[indices[j]]));
			count=count+1;
	
	# Make a list of pair(i,j) and corresponding distance tuples 
	dist_orig_tuple = [(pair[m],dist_orig[m]) for m in range(len(dist_orig))];
	dist_PCA256_tuple = [(pair[m],dist_PCA256[m]) for m in range(len(dist_orig))];
	dist_PCA512_tuple = [(pair[m],dist_PCA512[m]) for m in range(len(dist_orig))];
	dist_PCA1024_tuple = [(pair[m],dist_PCA1024[m]) for m in range(len(dist_orig))];
	dist_PCA2048_tuple = [(pair[m],dist_PCA2048[m]) for m in range(len(dist_orig))];
	dist_PCA4096_tuple = [(pair[m],dist_PCA4096[m]) for m in range(len(dist_orig))];
	
		
	# Sort the ((i,j),distance) tuples from low to high
	dist_orig_tuple.sort(key=lambda x: x[1], reverse=False);
	dist_PCA256_tuple.sort(key=lambda x: x[1], reverse=False);
	dist_PCA512_tuple.sort(key=lambda x: x[1], reverse=False);
	dist_PCA1024_tuple.sort(key=lambda x: x[1], reverse=False);
	dist_PCA2048_tuple.sort(key=lambda x: x[1], reverse=False);
	dist_PCA4096_tuple.sort(key=lambda x: x[1], reverse=False);
	
	
	pickle.dump(dist_orig_tuple,f1);
	pickle.dump(dist_PCA256_tuple,f2);
	pickle.dump(dist_PCA512_tuple,f3);
	pickle.dump(dist_PCA1024_tuple,f4);
	pickle.dump(dist_PCA2048_tuple,f5);
	pickle.dump(dist_PCA4096_tuple,f6);
	
	f1.close();
	f2.close();
	f3.close();
	f4.close();
	f5.close();
	f6.close();
	
		




