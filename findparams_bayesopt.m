function [lambda_1,maxiter,mv] = findparams_bayesopt(xTr,yTr,xVal,yVal,level_mark)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%OLD BAYESIAN OPTIMIZATION SCRIPT
addpath(genpath('bayesopt'));
%% Setting parameters for Bayesian Global Optimization
opt = defaultopt(); % Get some default values for non problem-specific options.
opt.dims = 2; % Number of parameters.
%%min/max for lambda_1  lambda_2   MAXITER
opt.mins =  [ 10, 50]; % Minimum value for each of the parameters. Should be 1-by-opt.dims
opt.maxes = [ 100, 200]; % Vector of maximum values for each parameter. 
opt.max_iters = 12; % How many parameter settings do you want to try?
opt.grid_size = 20000;
%opt=extractpars(varargin,opt);

%% Start the optimization
F = @(P) optimize_function(xTr,yTr,xVal,yVal,P,level_mark); % CBO needs a function handle whose sole parameter is a vector of the parameters to optimize over.
[bestP,mv] = bayesopt(F,opt);   % ms - Best parameter setting found
                               % mv - best function value for that setting L(ms)
                               % T  - Trace of all settings tried, their function values, and constraint values.


lambda_1=round(bestP(1));
%lambda_2=bestP(2);
maxiter=ceil(bestP(2));
fprintf('\n Best value:%2.4f',mv);
fprintf('\nBest parameters: lambda_1=%2.4f maxiter=%i!\n',lambda_1,maxiter);

end


function valerr=optimize_function(xTr,yTr,xVa,yVa,P,level_mark)
% function valerr=optimizeLMNN(xTr,yTr,xVa,yVa,P);
lambda_1=round(P(1));
%lambda_2=P(2);
maxiter=ceil(P(2));
fprintf('\nTrying lambda_1=%2.4f maxiter=%i!\n',lambda_1,maxiter);
 if(level_mark==1)
    [w,b]=SGD_hinge_normal(xTr,yTr,lambda_1,maxiter);
 else
    [w,b]=SGD_hinge_normal(xTr,yTr,lambda_1,maxiter);
 end
valerr=val_err(w,b,xVa,yVa);
%fprintf('validation error=%2.4f\n',valerr);

end