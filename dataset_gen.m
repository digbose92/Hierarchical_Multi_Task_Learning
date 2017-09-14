function [X,Y] = dataset_gen( label_set_child,train_cell,offset)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Y=[];
X=[];

for j=1:2
   label_curr=label_set_child{j};
   Y_temp=[];
   for k=1:size(label_curr,2)
    Y_temp=[Y_temp;train_cell{label_curr(k)}'];
   end
   
if(j==1)
Y=[Y;(-1)*ones(size(Y_temp,1),1)];
else
Y=[Y;(1)*ones(size(Y_temp,1),1)];
end
X=[X;Y_temp];
end

end

