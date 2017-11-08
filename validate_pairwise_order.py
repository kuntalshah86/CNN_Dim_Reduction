#This code is for validation of pair-wise distance oedering within the class
import numpy
import random

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
	
	f1=open("pairwise_dist_sorted_orig_dim_%s.dat" % label_dict_reverse[k],"w");
	f2=open("pairwise_dist_sorted_PCA256_%s.dat" % label_dict_reverse[k],"w");
	f3=open("pairwise_dist_sorted_PCA512_%s.dat" % label_dict_reverse[k],"w");
	f4=open("pairwise_dist_sorted_PCA1024_%s.dat" % label_dict_reverse[k],"w");
	f5=open("pairwise_dist_sorted_PCA2048_%s.dat" % label_dict_reverse[k],"w");
	f6=open("pairwise_dist_sorted_PCA4096_%s.dat" % label_dict_reverse[k],"w");
	
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
			dist_orig.append(squared_error_matrix_rows(X,indices[i],indices[j]));
			dist_PCA256.append(squared_error_matrix_rows(X_transformed_256,indices[i],indices[j]));
			dist_PCA512.append(squared_error_matrix_rows(X_transformed_512,indices[i],indices[j]));
			dist_PCA1024.append(squared_error_matrix_rows(X_transformed_1024,indices[i],indices[j]));
			dist_PCA2048.append(squared_error_matrix_rows(X_transformed_2048,indices[i],indices[j]));
			dist_PCA4096.append(squared_error_matrix_rows(X_transformed_4096,indices[i],indices[j]));
			count=count+1;
	
	# Make a list of pair(i,j) and corresponding distance tuples 
	dist_orig_tuple = [(pair[m],dist_orig[m]) for m in range(len(dist_orig))];
	dist_PCA256_tuple = [(pair[m],dist_PCA256[m]) for m in range(len(dist_orig))];
	dist_PCA512_tuple = [(pair[m],dist_PCA512[m]) for m in range(len(dist_orig))];
	dist_PCA1024_tuple = [(pair[m],dist_PCA1024[m]) for m in range(len(dist_orig))];
	dist_PCA2048_tuple = [(pair[m],dist_PCA2048[m]) for m in range(len(dist_orig))];
	dist_PCA4096_tuple = [(pair[m],dist_PCA4096[m]) for m in range(len(dist_orig))];
	
		
	# Sort the ((i,j),distance) tuples from high to low
	dist_orig_tuple.sort(key=lambda x: x[1], reverse=True);
	dist_PCA256_tuple.sort(key=lambda x: x[1], reverse=True);
	dist_PCA512_tuple.sort(key=lambda x: x[1], reverse=True);
	dist_PCA1024_tuple.sort(key=lambda x: x[1], reverse=True);
	dist_PCA2048_tuple.sort(key=lambda x: x[1], reverse=True);
	dist_PCA4096_tuple.sort(key=lambda x: x[1], reverse=True);
	
	for i in range(0,count):
		f1.write("%s %f\n" %(dist_orig_tuple[i][0],dist_orig_tuple[i][1]));
		f2.write("%s %f\n" %(dist_PCA256_tuple[i][0],dist_PCA256_tuple[i][1]));
		f3.write("%s %f\n" %(dist_PCA512_tuple[i][0],dist_PCA512_tuple[i][1]));
		f4.write("%s %f\n" %(dist_PCA1024_tuple[i][0],dist_PCA1024_tuple[i][1]));
		f5.write("%s %f\n" %(dist_PCA2048_tuple[i][0],dist_PCA2048_tuple[i][1]));
		f6.write("%s %f\n" %(dist_PCA4096_tuple[i][0],dist_PCA4096_tuple[i][1]));
	
	f1.close();
	f2.close();
	f3.close();
	f4.close();
	f5.close();
	f6.close();
	
		




#Python function to measure pair-wise squared error distances between 2 rows of a matrix
def squared_error_matrix_rows (matrix,row1,row2):
	return(numpy.sum(numpy.abs(matrix[row1,:]-matrix[row2,:])))