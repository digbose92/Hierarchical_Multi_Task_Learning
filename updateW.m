function [ W_curr_level ] = updateW( W_curr_level,WVal )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
k=1;
for j=1:size(W_curr_level,2)
    if(size(W_curr_level{j},1)>0)
      W_curr_level{j}=WVal(:,k);
      k=k+1;
    end

end

