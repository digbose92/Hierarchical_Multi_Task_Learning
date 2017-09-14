load('uci2v7.mat');         %uci digit 2 & 7
load('svmguide1-a.mat');   %svmguide1-a data (sampled 1000 points)
[n, dim] = size(x);

%k-means clustering
pval = kmeans(x,2); 
p = (pval-1.5)*2;
l = length(find(p~= label));
err1 = min(l, n-l)/n;

% iterative svr, initialized by k-means clustering
q = iterativeSVR(x, 500, 500, p, 0.1, 0.02);
l = length(find(q~= label));
err2 = min(l, n-l)/n;