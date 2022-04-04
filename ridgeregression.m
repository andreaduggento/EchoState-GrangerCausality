function [ iesn ] = ridgeregression( X, Y, esn)

	iesn = esn ;
	iesn.Wout = Y'*X'/(X*X'+esn.lambda*eye(size(X,1))); 

end

