function [ label_set_child ] = label_set_generate_itMMC(curr_node_data,curr_node_labels)
%UNTITLED3 Summary of this function goes here
%INPUT:
%   curr_node_data: the data in cell format
%   curr_node_labels=set of labels for the group
%OUTPUT:
%   label_set_child= a 2 member cell array with each element containing the
%   labels with the nodes


addpath(genpath('.\itMMC_code'));
%compute class specific mean of the class members in curr_node_data
num_class=size(curr_node_data,2);
mean_set=[];
for i=1:num_class
      mean_set=[mean_set,mean(curr_node_data{i},2)];
end

size(mean_set)
pval = kmeans(mean_set',2); 
p = (pval-1.5)*2;%rescaling the cluster indices to -1 and 1(for binary labels)
class(p)
[q,model_q] = iterativeSVR(mean_set', 500, 500, p, 0.1, 0.02);


q_positive=find(q==1);
q_negative=find(q==-1);

label_set_child={};
label_set_child{1}=curr_node_labels(q_positive);
label_set_child{2}=curr_node_labels(q_negative);



end


