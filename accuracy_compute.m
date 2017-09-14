function[acc_y_val]=accuracy_compute(label_val,level,num_class,validation_cell,id_l,child_num,W_level,C_level,offset_store_level,sz_val)

val_dat=[];
val_labels=[];

offset_val=zeros(1,num_class);

 
 
for i=1:num_class
if(i==1)
  offset_val(i)=0;
else
  offset_val(i)=offset_val(i-1)+sz_val(i-1);
end
end


for i=1:num_class
	val_dat=[val_dat,validation_cell{i}];
	val_labels=[val_labels,i*ones(1,sz_val(i))];
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

end

