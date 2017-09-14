function [rho,mv,values,samples] = bayesian_opt_param_select(Tr_cell,yTr,Val_cell,yVal,num_class)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%generates the best hyperparameter for the multi-task learning framework
addpath(genpath('bayesopt'));
%% Setting parameters for Bayesian Global Optimization
opt = defaultopt(); % Get some default values for non problem-specific options.
opt.dims = 1; % Number of parameters.
opt.mins =  [0.0000005]; % Minimum value for each of the parameters. Should be 1-by-opt.dims
opt.maxes = [0.0000006]; % Vector of maximum values for each parameter. 
opt.max_iters = 12; % How many parameter settings do you want to try?
opt.grid_size = 20000;

%% Start the optimization
F = @(P)validate_error_func(Tr_cell,yTr,Val_cell,yVal,P,num_class); % CBO needs a function handle whose sole parameter is a vector of the parameters to optimize over.
[bestP,mv,values,samples] = bayesopt(F,opt);   % ms - Best parameter setting found
                               % mv - best function value for that setting L(ms)
                               % T  - Trace of all settings tried, their function values, and constraint values.
rho=bestP(1);
fprintf('\n Best accuracy value:%2.4f',mv);
fprintf('\nBest parameters: rho=%2.10f\n',rho);

end

function val_error=validate_error_func(Tr_cell,yTr,Val_cell,yVa,P,num_class)
% function valerr=optimizeLMNN(xTr,yTr,xVa,yVa,P);
rho_1=P(1);
fprintf('\nTrying rho_1=%2.4f\n',rho_1);
[val_error]=tree_compute(Tr_cell,yTr,Val_cell,yVa,rho_1,num_class);
%fprintf('validation error=%2.4f\n',valerr);

end