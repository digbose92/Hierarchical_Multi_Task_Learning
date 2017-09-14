%test the working of lgmmc over the amazon dataset
%Data preprocessing for office dataset
num_class=31;
sz_tot=zeros(1,num_class);
sz_test=zeros(1,num_class);
addpath(genpath('C:\Users\bosed\Documents\vision_target\hierarchical_classification\LGMMC_v3'));
addpath(genpath('C:\Users\bosed\Documents\vision_target\hierarchical_classification\CPMMC'));
load('C:\Users\bosed\Documents\vision_target\hierarchical_classification\decaf7.mat');
addpath(genpath('C:\Users\bosed\Documents\vision_target\hierarchical_classification\itMMC_code'))
training_overall_set=[];
for i=1:num_class
        temp_train=train_cell{i}; %normalizing the columns of training data to unit norm
        %train_cell{i}=normc(temp_train);
        training_overall_set=[training_overall_set,train_cell{i}];
        sz_tot(i)=size(train_cell{i},2);
        temp_test=test_cell{i};
        %test_cell{i}=normc(temp_test); %normalizing the columns of test data to unit norm
        sz_test(i)=size(test_cell{i},2);
end
%train_cell{i}=contains the training data of the ith class
%test_cell{i}=contains the test data of the ith class
[sz_train,sz_val,offset,ind_train,ind_val]=train_test_split(training_overall_set,sz_tot,0.8,num_class);
train_cell=cell(1,num_class);
validation_cell=cell(1,num_class);
train_overall_set=[];
validation_overall_set=[];
for i=1:num_class
      train_cell{i}=training_overall_set(:,ind_train{i});
      train_overall_set=[train_overall_set,mean(train_cell{i},2)];
      validation_cell{i}=training_overall_set(:,ind_val{i});
      validation_overall_set=[validation_overall_set,mean(validation_cell{i},2)];
end
%train_overall_set(:,2)=[];
%opt.iteration=100;
%opt.C=0.0001;
%[pre_y,ct]= lgmmc(train_overall_set,opt);
%[lb,time]=CPMMC(train_overall_set);
pval = kmeans(train_overall_set',2); 
p = (pval-1.5)*2;
[q,model_q] = iterativeSVR(train_overall_set', 500, 500, p, 0.1, 0.02);
% 
minus_1=find(q==-1);
plus_1=find(q==1);
% 
% %second level split check
% plus_second_level=train_overall_set(:,plus_1);
% pval_second_level = kmeans(plus_second_level',2); 
% p_second = (pval_second_level-1.5)*2;
% q_second,model_q_second = iterativeSVR(plus_second_level', 500, 500, p_second, 0.1, 0.02);
% 
% 

