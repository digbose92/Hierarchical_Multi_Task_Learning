%main script for joint learning of the classifiers right from root to
%non-leaf node
%possible variant :- parent + two children joint training (like a triplet)
 
%joint training till root
%use entire data to learn the tree structure
%then split the data into validation and train to optimize the
%hyperparameters for the classifiers
%% READ THE DATA

%option=input('1.Enter the option for amazon dataset 2. Enter the option for willow dataset');
%amazon requires domain adaptation  
%willow doesnot require domain adaptation 
addpath('C:\Users\bosed\Documents\vision_target\Hierarchical_Multi_Task_Learning\itMMC_code');
addpath('C:\Users\bosed\Documents\vision_target\Hierarchical_Multi_Task_Learning\MALSAR1.1');
filepath='C:\Users\bosed\Documents\vision_target\Hierarchical_Multi_Task_Learning\mat_files\train_test_split_amazon_decaf.mat';
load(filepath);
%% TREE STRUCTURE LEARNING

%start with script and then proceed to function
unique_class_list=unique(train_labels);
num_class=size(unique_class_list,2);%listing the number of classes

sz_tot=zeros(1,num_class); %contains the sizes of each class coressponding to the training data
for i=1:num_class
   sz_curr_class=size(find(train_labels==i),2); 
   sz_tot(i)=sz_curr_class;  %sz_tot is the array containing the total size
end
train_tot_cell=cell(1,num_class);

offset_tot=zeros(1,num_class);

for i=1:num_class
   if(i==1)
      offset_tot(i)=0; 
   else
      offset_tot(i)=offset_tot(i-1)+sz_tot(i-1);
   end
   train_tot_cell{i}=train_set(:,offset_tot(i)+1:offset_tot(i)+sz_tot(i));
end


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

X_cell=cell({}); %cell holding all the data..X_cell{1}= data of the first task
Y_cell=cell({}); %
node_details=struct([]);
node_index=1;
%% JOINT LEARNING OF CLASSIFIER PARAMETERS (STARTING FROM ROOT)
while(leaf_num~=num_class)
  if(level==1)
     fprintf('\n Splitting the level:%d', level);
     label_acc=label_val{level}{1};
     [ label_set_child ] = label_set_generate_itMMC(train_tot_cell,label_acc);
      ntask=1;
      
      
     [Xcurr,Ycurr]=dataset_gen(label_set_child,train_tot_cell,0);
      X_cell{node_index}=Xcurr;
      Y_cell{node_index}=Ycurr;
      node_details(node_index).level=level;
      node_details(node_index).number=1;
   
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
     node_index=node_index+1;
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
        %X_curr_level=cell(1,n_l);
        Y_curr_level=cell(1,n_l);
        W_curr_level=cell(1,n_l);
        for j=1:n_l
            fprintf('\n Node:%d of level:%d is considered:',j,level);
            if(id_temp(j)==0)
                %non leaf node
                label_acc=label_val{level}{j};
                [train_cell_filter]=filter_train_data(train_tot_cell,label_acc);
                [label_set_child]=label_set_generate_itMMC(train_cell_filter,label_acc);
                %generating training data for the current node
                %[X_tr,Y_tr]=gen_train_data(train_tot_cell,label_set_child);
                %[X,Y]=dataset_gen(label_set_child,train_tot_cell);
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
                    [X,Y]=dataset_gen(label_set_child,train_tot_cell,lb);
                else      
                    lb=n_v_start+1;   
                    ub=n_v_start+2; 
                    [X,Y]=dataset_gen(label_set_child,train_tot_cell,lb-1);
                    for k=lb:ub %problem here for singular node i.e. when n_level=level+1;v=1
                        label_val{level+1}{k}=label_set_child{k-n_v_start};
                    end
                    n_v_start=n_v_start+2;      
                end 
                %X_curr_level{j}=X;
                %Y_curr_level{j}=Y;
                %W_curr_level{j}=rand(size(X,2),1);
                 X_cell{node_index}=X;
                 Y_cell{node_index}=Y;
                 node_details(node_index).level=level;
                 node_details(node_index).number=j;
                 node_index=node_index+1;
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
        
         
        
        node_level_mark{level+1}=node_labels;
        id_l{level+1}=id_level;
        leaf_num=leaf_num+sum(id_level); 
        % fprintf('\n Leaf number:%d',leaf_num);
        num_node{level+1}=sum(size_l); %total number of nodes in the next level
        child_num{level}=size_l;
    end
     level=level+1;   
end

%joint training together using L21 norm 
pho_1=0.00010;
opts.init = 0; % compute start point from data.
opts.tFlag = 1; % terminate after relative objective
% value does not changes much.
opts.tol = 10^-15; % tolerance.
opts.maxIter = 1500; % maximum iteration number of optimization
opts.rho_L2=0.005;
[W,c,fval]=Logistic_L21(X_cell,Y_cell,pho_1,opts);





%% ASSIGNMENT OF CLASSIFIER PARAMETERS TO THE NODES
%W_level=cell({}); %W_level{l} holds the weight vectors for the set of nodes per level
%C_level=cell({}); %C_level{l} holds the bias values for the set of nodes per level

[W_level,C_level]=assign_parameters(W,c,level,num_node,node_details);
%% TESTING


 
 