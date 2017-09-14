%checking the Subspace alignment code for decaf features currently used
num_class=31;
    sz_tot=zeros(1,num_class);
    sz_test=zeros(1,num_class);
    load('decaf7.mat');
    training_overall_set=[];
    train_labels=[];
    test_labels=[];
    test_overall_set=[];
    for i=1:num_class
        temp_train=train_cell{i}; %normalizing the columns of training data to unit norm
        %train_cell{i}=normc(temp_train);
        training_overall_set=[training_overall_set,train_cell{i}];
        train_labels=[train_labels,i*ones(1,size(train_cell{i},2))];
        test_labels=[test_labels,i*ones(1,size(test_cell{i},2))];
        test_overall_set=[test_overall_set,test_cell{i}];
        sz_tot(i)=size(train_cell{i},2);
        temp_test=test_cell{i};
        %test_cell{i}=normc(temp_test); %normalizing the columns of test data to unit norm
        sz_test(i)=size(test_cell{i},2);
    end
    training_overall_set=training_overall_set';
    test_overall_set=test_overall_set';
    addpath(genpath('.\DA_SA\DA_SA'));
    
    training_overall_set = training_overall_set ./ repmat(sum(training_overall_set,2),1,size(training_overall_set,2));  
    test_overall_set = test_overall_set ./ repmat(sum(test_overall_set,2),1,size(test_overall_set,2));  
    training_overall_set=zscore(training_overall_set);
    test_overall_set=zscore(test_overall_set);
    
    subspace_dim_d = 80;
    [Xss,~,~] = princomp(training_overall_set);
    [Xtt,~,~]=princomp(test_overall_set);
    Xs = Xss(:,1:subspace_dim_d);
    Xt = Xtt(:,1:subspace_dim_d);
    [accuracy_na_nn,accuracy_sa_nn] = Subspace_Alignment(training_overall_set,test_overall_set,train_labels,test_labels,Xs,Xt);

    %[Xtt,~,~] = pca(Target);
    
    
    
    
    
   %[sz_train,sz_val,offset,ind_train,ind_val]=train_test_split(training_overall_set,sz_tot,0.8,num_class);
   %train_cell=cell(1,num_class);
   %validation_cell=cell(1,num_class);
   %train_data=[];
   %train_label=[];
   %for i=1:num_class
    %  train_cell{i}=training_overall_set(:,ind_train{i});
    %  train_data=[train_data,train_cell{i}];
    %  train_label=[train_label,i*ones(1,sz_train(i))];
    %  validation_cell{i}=training_overall_set(:,ind_val{i});
  % end 
  