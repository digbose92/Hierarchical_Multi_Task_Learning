function [W_level,C_level] = assign_parameters(W,C,level,num_node,node_details)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
W_level=cell({});
C_level=cell({});

%W_level is the level wise cell array 
for l=1:level
    n_l=num_node{l}(1);
    W_level{l}=cell(1,n_l); %initialization done 
    C_level{l}=cell(1,n_l);
end

size(node_details)
W_level
C_level
%loop over node details
for index=1:size(node_details,2)
   W_level{node_details(index).level}{node_details(index).number}=W(:,index);
   C_level{node_details(index).level}{node_details(index).number}=C(index); 
end


