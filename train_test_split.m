function [sz_train,sz_test,offset,ind_train,ind_test] = train_test_split(in,sz,M,num_class)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


%% ======INPUT DESCRIPTION==============
% in :input data
% sz: size of the input data with each entry of sz denoting the size of a class
% M: the fraction of data in training data (1-M in test data) 
% num_class: number of classes
%% ==== OUTPUT DESCRIPTION================
% train_data: the training set
% test_data: the test set
% sz_train: the size of the training set (sz_train(i)=size of the training data of class i)
% sz_test:  the size of the test set(sz_test(i)=size of the test data of class i)
%%
sz_train=round(M.*sz);
sz_test=sz-sz_train;
ind_train=cell(1,num_class);
ind_test=cell(1,num_class);

%%ind{i} here refers to the indices coressponding to the training data
%%associated with  class i
%offset=[0,230,389,523,596,664,857,1060];
offset=zeros(1,num_class);
for i=2:num_class
    
    offset(i)=offset(i-1)+sz(i-1);
end
for i=1:num_class
    ind_train{i}=offset(i)+randperm(sz(i),sz_train(i));
    ind_train{i}=sort(ind_train{i},'ascend');
end
ind_check=cell(1,num_class);
%%find indices coressponding to the test data associated with class i 
temp_ind=[];
temp_test=[];
for i=1:num_class
    ind_temp=zeros(1,sz(i));
   temp_ind=ind_train{i};
   for k=1:sz_train(i)
       for j=offset(i)+1:offset(i)+sz(i)
           if(temp_ind(k)==j)
               ind_temp(j-offset(i))=1;
               break;
           end
       end
   end
   ind_check{i}=ind_temp;
end
% 
 for i=1:num_class
    tmp=ind_check{i};
    tp_data=[];
   for k=1:sz(i) 
        if(tmp(k)==0)
            tp_data=[tp_data offset(i)+k];
    end
    end
  ind_test{i}=tp_data;
 end
train_data=[];
test_data=[];
for i=1:num_class
    id_train=ind_train{i};
    id_test=ind_test{i};
    tr_temp=in(:,id_train);
    test_temp=in(:,id_test);
    train_data=[train_data,tr_temp];
    test_data=[test_data,test_temp];
end
end

