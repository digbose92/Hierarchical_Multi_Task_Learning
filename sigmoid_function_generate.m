function [class]=sigmoid_function_generate(W,b,x)

	score=W'*x+b;
	value=1/(1+exp(-score));
	if(value >=0.5)
		class=1;
	else
		class=-1;
	end
	