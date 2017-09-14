function [w,b,lambda_1,maxiter,mv] = train_overall( tr_cell,val_cell,label_set_child,level_mark)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    [X_tr,Y_tr]=gen_train_data(tr_cell,label_set_child); 
    [X_val,Y_val]=gen_train_data(val_cell,label_set_child); 
    [lambda_1,maxiter,mv]=findparams_bayesopt(X_tr,Y_tr,X_val,Y_val,level_mark);
    %now overall training 
   %aggregate the training and validation data
    train_data_overall=[];
    label_data_overall=[];
    num_class=size(1,size(label_set_child,2));
    for i=1:num_class
       label_set_curr=label_set_child{i};
       train_temp=[];
       for j=1:size(label_set_curr,2)
           train_temp=[train_temp,tr_cell{label_set_curr(j)},val_cell{label_set_curr(j)}];
           
       end
       train_data_overall=[train_data_overall,train_temp];
       if(i==1)
            label_data_overall=[label_data_overall,1*ones(1,size(train_temp,2))];
       else
            label_data_overall=[label_data_overall,-1*ones(1,size(train_temp,2))];
       end
   end
if(level_mark==1) 
   
   [w,b]=SGD_hinge_normal(train_data_overall,label_data_overall,lambda_1,maxiter);
   
else
    %node of the other level
    [w,b]=SGD_hinge_normal(train_data_overall,label_data_overall,lambda_1,maxiter);
    
end
end

