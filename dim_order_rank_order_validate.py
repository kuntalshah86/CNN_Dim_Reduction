import numpy

classes=['paisley','plain','floral','vertical','horizontal','diagonal','spotted','gingham'];
label_dict=dict(zip(classes,range(len(classes))));
label_dict_reverse=dict(zip(range(len(classes)),classes));

#Top n pairs with least distance to check
n=100;
avg_rank_change_mat=numpy.zeros((1,6));

#for k in [0,3,5,6,7]:
for k in [0,5,6]:
	D_orig=numpy.load("pairwise_dist_sorted_orig_dim_%s.dat" % label_dict_reverse[k]);
	D_PCA256=numpy.load("pairwise_dist_sorted_PCA256_%s.dat" % label_dict_reverse[k]);
	D_PCA512=numpy.load("pairwise_dist_sorted_PCA512_%s.dat" % label_dict_reverse[k]);
	D_PCA1024=numpy.load("pairwise_dist_sorted_PCA1024_%s.dat" % label_dict_reverse[k]);
	D_PCA2048=numpy.load("pairwise_dist_sorted_PCA2048_%s.dat" % label_dict_reverse[k]);
	D_PCA4096=numpy.load("pairwise_dist_sorted_PCA4096_%s.dat" % label_dict_reverse[k]);
	
	cum_rank_diff_PCA256=0;
	cum_rank_diff_PCA512=0;
	cum_rank_diff_PCA1024=0;
	cum_rank_diff_PCA2048=0;
	cum_rank_diff_PCA4096=0;
		
	for i in range(0,max(len(D_orig),n)):
		index_orig=i;
		index_PCA256= [index for index in range(len(D_orig)) if D_PCA256[index][0][0] == D_orig[i][0][0] and D_PCA256[index][0][1] == D_orig[i][0][1]][0];
		index_PCA512= [index for index in range(len(D_orig)) if D_PCA512[index][0][0] == D_orig[i][0][0] and D_PCA512[index][0][1] == D_orig[i][0][1]][0];
		index_PCA1024= [index for index in range(len(D_orig)) if D_PCA1024[index][0][0] == D_orig[i][0][0] and D_PCA1024[index][0][1] == D_orig[i][0][1]][0];
		index_PCA2048= [index for index in range(len(D_orig)) if D_PCA2048[index][0][0] == D_orig[i][0][0] and D_PCA2048[index][0][1] == D_orig[i][0][1]][0];
		index_PCA4096= [index for index in range(len(D_orig)) if D_PCA4096[index][0][0] == D_orig[i][0][0] and D_PCA4096[index][0][1] == D_orig[i][0][1]][0];
		
		cum_rank_diff_PCA256=cum_rank_diff_PCA256+abs(index_PCA256-index_orig);
		cum_rank_diff_PCA512=cum_rank_diff_PCA512+abs(index_PCA512-index_orig);
		cum_rank_diff_PCA1024=cum_rank_diff_PCA1024+abs(index_PCA1024-index_orig);
		cum_rank_diff_PCA2048=cum_rank_diff_PCA2048+abs(index_PCA2048-index_orig);
		cum_rank_diff_PCA4096=cum_rank_diff_PCA4096+abs(index_PCA4096-index_orig);
		
	avg_rank_change_PCA256=cum_rank_diff_PCA256/max(len(D_orig),n);
	avg_rank_change_PCA512=cum_rank_diff_PCA512/max(len(D_orig),n);
	avg_rank_change_PCA1024=cum_rank_diff_PCA1024/max(len(D_orig),n);
	avg_rank_change_PCA2048=cum_rank_diff_PCA2048/max(len(D_orig),n);
	avg_rank_change_PCA4096=cum_rank_diff_PCA4096/max(len(D_orig),n);
	
	print ("%s %f %f %f %f %f" %(label_dict_reverse,avg_rank_change_PCA256,avg_rank_change_PCA512,avg_rank_change_PCA1024,avg_rank_change_PCA2048,avg_rank_change_PCA4096));
	
	
	

