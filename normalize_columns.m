function [X_normalize_set] = normalize_columns( X )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
X_normalize_set=zeros(size(X));

for i=1:size(X,2)
   X_normalize_set(:,i)=X(:,i)./norm(X(:,i));
end

end

