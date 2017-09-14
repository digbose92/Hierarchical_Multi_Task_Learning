%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iterative SVR algorithm for binary clustering        %
% x:      N by D data matrix                           %
% sigma:  RBF kernel width K(x)=exp(|x|^2/sigma^2)     %
% C:      regularization parameter                     %
% y:      initial labels for iterSVR to start          %
%         (usually chosen as k-means clusterign result)%
% epsi:   epsilon-sensitive loss in SVR                %
%         theoretically, the smaller, the better       %
%         in practice may be chosen in domain [0,0.2]  %
% ell:    balance parameter, a ratio in (0, 0.5)       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y,model] = iterativeSVR(x, sigma, C, y, epsi, ell);
[n, dim] = size(x);
gamma = 1/sigma^2;

st= zeros(1,50);
for ite = 1:50;   

    opt = sprintf('-s 3 -t 2 -g %g -c %g -p %g -e 0.001', gamma, C, epsi);
    model = svmtrain(y, x, opt);
    %w=(model.sv_coef);
    %print(w)
    [pdt, acc, dec_values] = svmpredict(y, x, model); 
    
    bias = cal_bias_SVR(pdt, n, ell);
    pdt = pdt - bias;
    
    y(find(pdt>=0)) = 1;
    y(find(pdt<0)) =-1;
    st(ite) = norm(y - pdt);
    if(ite>1 & abs( st(ite) - st(ite-1)) <= 1e-3 * st(ite - 1));
        break;
    end;
end;