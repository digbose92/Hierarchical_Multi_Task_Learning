clc;
clear all;
%%MAIN VALIDATION SCRIPT
filepath='C:\Users\bosed\Documents\vision_target\hierarchical_classification\hierarchical_itMMC_codes\mat_files\train_test_split_amazon_decaf.mat';
load(filepath);

%later insert a choice for retraining using the best parameters available
%on the entire train+validation combination


%% split overall_training data to train_data and validation_data
unique_class_list=unique(train_labels);
num_class=size(unique_class_list,2);%listing the number of classes

sz_tot=zeros(1,num_class); %contains the sizes of each class coressponding to the training data
for i=1:num_class
   sz_curr_class=size(find(train_labels==i),2); 
   sz_tot(i)=sz_curr_class;
end
[sz_train,sz_val,offset,ind_train,ind_val]=train_test_split(train_set,sz_tot,0.8,num_class); %80-20 split for train/validation


%%  generate the training and validation cells
train_cell=cell(1,num_class);
validation_cell=cell(1,num_class);
act_train_label=[];
act_val_label=[];
for i=1:num_class
      train_cell{i}=train_set(:,ind_train{i});
      act_train_label=[act_train_label,i*ones(1,sz_train(i))];
      act_val_label=[act_val_label,i*ones(1,sz_val(i))];
      validation_cell{i}=train_set(:,ind_val{i});
end

%% call the bayesian optimization routine
[rho,bestval,values,samples] = bayesian_opt_param_select(train_cell,act_train_label,validation_cell,act_val_label,num_class);

%rho is the best parameter to be returned
%bestval is the coressponding validation accuracy
fprintf('\n Best parameter:%2.4f, Best accuracy value:%2.4f',rho,(100-bestval));

%% save the best parameters
fid = fopen('./validation_results/bayesian_results.txt','wt');
fprintf(fid,'Param:%0.10f Best accuracy:%0.6f\n',rho,(100-bestval));
fclose(fid);

%% run on the test set and list the final accuracy




