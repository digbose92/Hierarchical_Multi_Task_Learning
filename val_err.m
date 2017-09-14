function [validation_error] = val_err(w,b,xVa,yVa)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
ypred=[];
for i=1:size(xVa,2)
    score=(w'*xVa(:,i))+b;
    if(score>0)
         ypred=[ypred,1];
    elseif(score<0)
        ypred=[ypred,-1];
    else
        ypred=[ypred,1];
    end
end
validation_error=sum((yVa~=ypred))/size(yVa,2);
fprintf('val_error:%2.4f',validation_error);
end

