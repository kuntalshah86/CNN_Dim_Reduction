import codecs
import pickle
import glob
import os
import re
import numpy
import matplotlib
import math

#Preparing the data

classes=['paisley','plain','floral','vertical','horizontal','diagonal','spotted','gingham'];
label_dict=dict(zip(classes,range(len(classes))));
combined_regex = "(" + ")|(".join(classes) + ")";
X=numpy.matrix(range(8192),dtype=numpy.float32);
y=numpy.matrix(range(1),dtype=numpy.float32);
mean=numpy.matrix(range(8192),dtype=numpy.float32);

#reduced dimensions, we want to test 256,512,1024,2048,4096 dimensional feature space post PCA
d=[256,512,1024,2048,4096];

for f in glob.glob('*.pkl'):
	match = re.search(combined_regex,os.path.basename(f));
	if not match:
		continue
		
	f=codecs.open('%s' % f,mode='rb', encoding=None, errors='strict', buffering=1);
	dictionary=pickle.load(f,encoding='iso-8859-1');
	for i in range(0,len(dictionary.keys()),1):
					
		X=numpy.vstack((X,dictionary[list(dictionary)[i]]['Features_1']));
		y=numpy.vstack((y,label_dict[match.group()]));

#Delete first row of matrices
X=numpy.delete(X,0,axis=0);
y=numpy.delete(y,0,axis=0);		

#Get the mean vector, if calculating scatter matrix ourselves
#PCA - mean is per feature across all the classes
#Skipping this for now as the original data seems to be mean normalized and scaled.
#Need to add this back in when we scale to more data
#mean=numpy.array(range(8192),dtype=numpy.float64);
#for i in range(1,X.shape[1]):
#	mean[i]=float(numpy.mean(X[:,i]));
	
#calculate the covariance matrix 
cov_mat=numpy.dot(X.T,X);

#Calculate eigenvalues and eigenvectors

eig_val_cov, eig_vec_cov = numpy.linalg.eig(cov_mat);


#Checking eigenvectors are unit vectors
#This is just for validation, commenting in the production code
#for ev in eig_vec_cov:
#    numpy.testing.assert_array_almost_equal(2.0,numpy.linalg.norm(ev))
		
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(numpy.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]	
		
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
#This is just for validation, commenting in the production code
#for i in eig_pairs:
#    print(i[0])

#Dump out W matrix and trasformed feature matrix for lower dimensional spaces of different dimensions
for k in d:
    W=numpy.zeros((X.shape[1],1),dtype=numpy.float64)
    for i in range(0,k,1):
        W=numpy.hstack((W,eig_pairs[i][1].reshape(X.shape[1],1)))
    W=numpy.delete(W,0,axis=1)
    X_transformed=numpy.matrix((W.T.dot(X.T)).T);
    filename1 = "W_transform_matrix" + str( k ) + ".dat";
    filename2 = "X_transformed_feature_dimension_" + str( k ) + ".dat";
    W.dump(filename1);
    X_transformed.dump(filename2);
    