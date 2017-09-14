function [ label_set_child ] = child_labels_set( label_set,index_child)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%label_set=set of labels of the parent
%label_set_child=set of exact labels of the childs
label_set_child=cell(1,2);
for i=1:2
    index_set_temp=index_child{i};
    label_set_child{i}=label_set(index_set_temp);
end

end

