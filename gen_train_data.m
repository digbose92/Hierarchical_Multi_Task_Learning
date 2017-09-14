function [ train_data, label ] = gen_train_data( train_cell,label_set_child )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
num_class=size(label_set_child,2);
train_data=[];
label=[];
train_class_cell=cell(1,num_class);
for i=1:num_class
    label_current=label_set_child{i};
    train_temp=[];
    for j=1:size(label_current,2)
        train_temp=[train_temp,train_cell{label_current(j)}];
    end
    train_data=[train_data,train_temp];
    %train_class_cell{i}=train_temp;
    if(i==1)
        label=[label,1*ones(1,size(train_temp,2))];
    else
        label=[label,-1*ones(1,size(train_temp,2))];
    end
    
end

