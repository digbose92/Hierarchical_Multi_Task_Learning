function [ id ] = leaf_check(label_set_child,num_child )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
id=zeros(1,num_child);
%id(i)=0 means the ith child is not a leaf
%id(i)=1 means the ith child is a leaf
for i=1:num_child
    tmp_lb=label_set_child{i};
    if(size(tmp_lb,2)==1) % if there is only one label
        id(i)=1;
    else
        id(i)=0;   
    end

end

