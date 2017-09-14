%checking current split of data for results
clc;
clear all;
file_path='C:\Users\bosed\Documents\vision_target\comparison_algo\curr_split.mat';

load(file_path);

num_class=31;

sz_train=[];
sz_val=[];
for i=1:num_class
    curr_Size_train=find(train_label==i);
    curr_Size_val=find(val_labels==i);
    sz_train=[sz_train,size(curr_Size_train,2)];
    sz_val=[sz_val,size(curr_Size_val,2)];
end

offset_train=zeros(1,num_class);

for i =1:num_class
   if(i==1)
       offset_train(i)=0;
   else
       offset_train(i)=offset_train(i-1)+sz_train(i-1);
   end
end


%generate the training cell data
train_cell=cell(1,num_class);
validation_cell=cell(1,num_class);

for i=1:num_class
      train_cell{i}=train_data(:,offset_train(i)+1:offset_train(i)+sz_train(i));
end 

acc_list=[];
%rho_list=[0.0005,0.005,0.06,0.05,0.04,0.03,0.02,0.01,0.1,0.2,0.3,0.4,0.5];
rho_list=linspace(0.05,0.06,10);
%rho_list=[0.5,0.6,0.7,0.8,0.9,1];
for i=1:size(rho_list,2)
rho_1=rho_list(i);
leaf_num=0;
level=1; %level of the root
num_node{level}=1; %number of nodes at level 1
id_l=cell({}); %id_l{lvl} is an array at level lvl with number of entries equal to number of nodes in the tree
child_num=cell({});
id_l{level}=0;
label_val{level}{1}=1:num_class;
node_level_mark=cell({});
node_level_mark{1}=1; % first level indicates root 
node_mark=[1];
node_labels=[];
Parent_set=[0];
id_tot=[0];
parent_level_mark=cell({});

load('C:\Users\bosed\Documents\vision_target\hierarchical_classification\MALSAR1.1\data\school.mat'); %loading data - X and Y are two cell arrays with the number of members equal to number of tasks
addpath(genpath('C:\Users\bosed\Documents\vision_target\hierarchical_classification\MALSAR1.1\MALSAR\functions\Lasso'));



% FOLLOWING TAKEN FROM THE LEAST LASSO EXAMPLE
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-15;   % tolerance. 
opts.maxIter = 1500; % maximum iteration number of optimization.
W_level=cell({});
f_level=cell({});
C_level=cell({});
Y_level=cell({});
%w_cell=cell({});
%b_cell=cell({});
%val_error=cell({});
%[ label_set_child ] = label_set_generate_itMMC(train_cell,label_val{1}{1});
while(leaf_num~=num_class)
  if(level==1)
     fprintf('\n Splitting the level:%d', level);
     label_acc=label_val{level}{1};
     [ label_set_child ] = label_set_generate_itMMC(train_cell,label_acc);
      ntask=1;
      X=cell(1,ntask);Y=cell(1,ntask);
      
     [Xcurr,Ycurr]=dataset_gen(label_set_child,train_cell,0);
      X{1}=Xcurr;
      Y{1}=Ycurr;
      
     [W, C, funcVal,fval] = Logistic_Lasso(X, Y, rho_1, opts);
     W_curr_level{1}=W;
     W_level{1}=W_curr_level; %per level storing the Weight cell of the level
     C_level{1}=C;
     f_level{1}=fval;
     Y_level{1}=Y;
     num_node{level+1}=2;
     child_num{level}=2; %number of child nodes associated with the node at level l
     node_level_mark{level+1}=[node_mark(end)+1:node_mark(end)+2];% marking of nodes in current level
     node_labels=[node_labels,node_mark(end)+1:node_mark(end)+2];
     node_mark=[node_mark,node_mark(end)+1:node_mark(end)+2];%marking of the node
     Parent_set=[Parent_set,(Parent_set(end)+1)*ones(1,2)];
     [id]=leaf_check(label_set_child,2);
     id_tot=[id_tot,id];
     id_l{level+1}=id;
     for j=1:2
             label_val{level+1}{j}=label_set_child{j};
     end
  else
      n_l=num_node{level}; %number of nodes in the level l = number of child nodes of nodes at level l-1
        id_temp=id_l{level};
        fprintf('\n Training the level:%d', level);
        node_mark_parent_set=node_level_mark{level};
        size_l=zeros(1,n_l);%size_l(i) determines the number of child nodes associated with the ith node at level l
        %in case of root it was a single number because there was a single
        %node
        %in case size_l(i)=0 means the particular ith node is a leaf
        id_level=[];
        node_labels=[];
        X_curr_level=cell(1,n_l);
        Y_curr_level=cell(1,n_l);
        W_curr_level=cell(1,n_l);
        for j=1:n_l
            fprintf('\n Node:%d of level:%d is considered:',j,level);
            if(id_temp(j)==0)
                %non leaf node
                label_acc=label_val{level}{j};
                [train_cell_filter]=filter_train_data(train_cell,label_acc);
                [label_set_child]=label_set_generate_itMMC(train_cell_filter,label_acc);
                %generating training data for the current node
                %[X_tr,Y_tr]=gen_train_data(train_cell,label_set_child);
                %[X,Y]=dataset_gen(label_set_child,train_cell);
                %X_curr_level{j}=X;
                %Y_curr_level{j}=Y;
                %W_curr_level{j}=rand(size(X,2),1);
                node_labels=[node_labels,node_mark(end)+1:node_mark(end)+2];
                node_mark=[node_mark,node_mark(end)+1:node_mark(end)+2];
                Parent_set=[Parent_set, (node_mark_parent_set(j))*ones(1,2)];
                size_l(j)=2;
                [id]=leaf_check(label_set_child,2); %for the jth node at the lth level
                id_level=[id_level,id];
                id_tot=[id_tot,id];
                if(j==1) %j=1 coressponds to the first node at current level
                    for k=1:2
                        label_val{level+1}{k}=label_set_child{k};
                    end
                    lb=0;
                    n_v_start=2;    
                    [X,Y]=dataset_gen(label_set_child,train_cell,lb);
                else      
                    lb=n_v_start+1;   
                    ub=n_v_start+2; 
                    [X,Y]=dataset_gen(label_set_child,train_cell,lb-1);
                    for k=lb:ub %problem here for singular node i.e. when n_level=level+1;v=1
                        label_val{level+1}{k}=label_set_child{k-n_v_start};
                    end
                    n_v_start=n_v_start+2;      
                end 
                X_curr_level{j}=X;
                Y_curr_level{j}=Y;
                W_curr_level{j}=rand(size(X,2),1);
            else
                X_curr_level{j}=[];
                Y_curr_level{j}=[];
                W_curr_level{j}=[];
                %leaf node
                fprintf(' Leaf');
                
                %leaf node
                if(j==1)
                    n_v_start=0;
                else
                    n_v_start=n_v_start;
                    
                end
                
            end
        end
        
          X_cell_non_empty= X_curr_level(~cellfun(@isempty, X_curr_level));
          Y_cell_non_empty=Y_curr_level(~cellfun(@isempty, Y_curr_level));
          [WVal, CVal, funcVal,fval] = Logistic_Lasso(X_cell_non_empty, Y_cell_non_empty, rho_1, opts);
           
          [W_curr_level]=updateW(W_curr_level,WVal);
          W_level{level}=W_curr_level;
          f_level{level}=fval;
          C_level{level}=CVal;
          Y_level{level}=Y_curr_level;
        
        
        node_level_mark{level+1}=node_labels;
        id_l{level+1}=id_level;
        leaf_num=leaf_num+sum(id_level); 
        % fprintf('\n Leaf number:%d',leaf_num);
        num_node{level+1}=sum(size_l); %total number of nodes in the next level
        child_num{level}=size_l;
    end
     level=level+1;
   
     
       
       
end
   


%results on the validation set
%offset generation
for l=1:level-1
     n_node=num_node{l};
     id_level=id_l{l};
     offset_temp=zeros(1,n_node);
     for k=1:n_node
         if(id_level(k)==0)
         if (k==1)
          offset_temp(k)=0;
         elseif(id_level(k-1)==1)
         offset_temp(k)=offset_temp(k-1);
         else
           
             offset_temp(k)=child_num{l}(k-1)+offset_temp(k-1);
         end
         else
             if(k==1)
                 offset_temp(k)=0;
             else
             offset_temp(k)=offset_temp(k-1)+child_num{l}(k-1);
             end
         end
         
         
     end
     offset_store_level{l}=offset_temp;
end 


offset_val=zeros(1,num_class);

 
 
for i=1:num_class
if(i==1)
  offset_val(i)=0;
else
  offset_val(i)=offset_val(i-1)+sz_val(i-1);
end
end



%yval holds the predicted labels for the test samples
yval=zeros(1,size(val_dat,2));

test_start_time=clock();
for index_val=1:size(val_dat,2)
	fprintf('\n Test sample_%d:',index_val);
	tvect=val_dat(:,index_val); 
	for l=1:level
	id_node_ar=id_l{l};
	if(l==1)
		num_class_temp=child_num{l}(1); 
		W_curr_node=W_level{l}{1};
		C_curr_node=C_level{l}(1);
		class_gen=sigmoid_function_generate(W_curr_node,C_curr_node,tvect);
		offset_curr_node=offset_store_level{l}(1);
		[next_ind_val]=node_num_gen(class_gen,l,offset_curr_node);
		
	else
		if(id_node_ar(next_ind_val)==1) %leaf node        
            p_lb=label_val{l}{next_ind_val};
             yval(index_val)=p_lb;
		break;
		else 
			W_curr_node=W_level{l}{next_ind_val};   %L for the root node
			C_curr_node=C_level{l}(next_ind_val); 
			class_gen=sigmoid_function_generate(W_curr_node,C_curr_node,tvect);
			offset_curr_node=offset_store_level{l}(next_ind_val);
			[next_ind_val]=node_num_gen(class_gen,l,offset_curr_node);
		end
		
	end
	end
end

time_validate=etime(clock,test_start_time);
fprintf('\n Time elapsed in validation: %d', time_validate);


denum=size(val_labels,2);
s_val=sum(yval==val_labels);
acc_y_val=((s_val)/(denum))*100;
fprintf('\n Accuracy in validation using hierarchical multi task MMC method : %d', acc_y_val);
acc_list=[acc_list,acc_y_val];
end