%checking Logistic Lasso

load('C:\Users\bosed\Documents\vision_target\hierarchical_classification\MALSAR1.1\data\school.mat'); %loading data - X and Y are two cell arrays with the number of members equal to number of tasks
addpath(genpath('C:\Users\bosed\Documents\vision_target\hierarchical_classification\MALSAR1.1\MALSAR\functions\Lasso'));

rho_1=20;

% FOLLOWING TAKEN FROM THE LEAST LASSO EXAMPLE
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1500; % maximum iteration number of optimization.

[W, C, funcVal,fval] = Logistic_Lasso(X, Y, rho_1, opts);