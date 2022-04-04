function  [ iesn ] = elasticridgeregression( X, Y, esn)

	iesn = esn ;
	s = warning('error','MATLAB:nearlySingularMatrix');
	retry=1;
	lambda = esn.lambda;

	if lambda == 0.
		W = Y'*X'/(X*X'+lambda*eye(size(X,1)));
	else
		while retry>0
			retry=0;
			try	
				W = Y'*X'/(X*X'+lambda*eye(size(X,1)));
			catch
				retry=1;
				lambda = lambda*2.;
			end
		end
	end

	iesn.lambda = lambda;
	iesn.Wout = W;

	warning(s);
end

