function [ tr_filter_cell,num_class_set ] = filter_train_data( tr_cell,label_set )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
num_class_set=size(label_set,2);
tr_filter_cell=cell(1,num_class_set);
for i=1:size(label_set,2)
    tr_filter_cell{i}=tr_cell{label_set(i)};
end

end

