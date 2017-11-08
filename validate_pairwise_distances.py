#This code is for validation
import numpy
import random

#Initialize number of pairs to test per class
num_pairs=10000;

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

for i in sorted(label_dict.values()):
	indices = [index for index in range(len(y)) if y.item(index) == i];
	count=0;
	while(count<num_pairs):
		k=random.sample(indices, 2);
		#Dump output to a file
		print(label_dict_reverse[i],k[0],k[1],squared_error_matrix_row(X,k[0],k[1]),squared_error_matrix_row(X_transformed_256,k[0],k[1]),squared_error_matrix_row(X_transformed_512,k[0],k[1]),squared_error_matrix_row(X_transformed_1024,k[0],k[1]),squared_error_matrix_row(X_transformed_2048,k[0],k[1]),squared_error_matrix_row(X_transformed_4096,k[0],k[1]));	
		count=count+1;




#Python function to measure pair-wise squared error distances between 2 rows of a matrix
def squared_error_matrix_rows (matrix,row1,row2):
	return(numpy.sum(numpy.abs(matrix[row1,:]-matrix[row2,:])))