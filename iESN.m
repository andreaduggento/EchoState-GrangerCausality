%%%%%	This is a modified implementation initially taken from https://github.com/stefanonardo/echo-state-network/

classdef iESN < handle
    properties
        Nr 
        Ntot 
        alpha
        rho
        inputScaling
        biasScaling
        lambda
        connectivity
        readout_training
        orthonormalWeights
	seqDim
        Win
        Wb
        Wr
        Wout
        internalState
	nonlinearfunction    
    end
    methods
        function iesn = iESN(Nr, seqDim, varargin)
        	% Constructor
        	%
        	% args:
        	%   Nr: reservoir's size
        	%   
        	% optional args:
        	%   'leakRate': leakage rate
        	%   'spectralRadius': spectral radius
        	%   'inputScaling': input weights scale 
        	%	'biasScaling': bias weights scale 
        	%	'regularization': regularization parameter
        	%	'connectivity': reservoir connectivity
        	%	'readoutTraining': readout training method
        	    
        	iesn.Nr = Nr;
		iesn.Ntot=sum(Nr);
        	iesn.alpha = 1;
        	iesn.rho = 0.9;
        	iesn.inputScaling = 1;
        	iesn.biasScaling = 1;
        	iesn.lambda = 1;
        	iesn.connectivity = 1;
        	iesn.readout_training = 'ridgeregression';
        	iesn.orthonormalWeights = 1;
        	iesn.nonlinearfunction = 'tanh';

        	numvarargs = length(varargin);
        	for i = 1:2:numvarargs
        	    switch varargin{i}
        	        case 'leakRate', iesn.alpha = varargin{i+1};
        	        case 'spectralRadius', iesn.rho = varargin{i+1};
        	        case 'inputScaling', iesn.inputScaling = varargin{i+1};
        	        case 'biasScaling', iesn.biasScaling = varargin{i+1};
        	        case 'regularization', iesn.lambda = varargin{i+1};
        	        case 'connectivity', iesn.connectivity = varargin{i+1};
        	        case 'readoutTraining', iesn.readout_training = varargin{i+1};
        	        case 'orthonormalWeights', iesn.orthonormalWeights = varargin{i+1};
        	        case 'nonlinearfunction', iesn.nonlinearfunction = varargin{i+1};
        	        case 'Wr', iesn.Wr = varargin{i+1};
        	        
        	        otherwise, error('the option does not exist');
        	    end
        	end
        
        	if( seqDim ~= size(Nr,2))
                	error('seqDim mismatch');
		end

         	iesn.seqDim = seqDim;
        	iesn.Win = [] ; for i=1:seqDim ; wtemp = iesn.inputScaling*(rand(Nr(i),1) * 2 - 1); iesn.Win=blkdiag(iesn.Win,wtemp); end;
		iesn.Wb = iesn.biasScaling * (rand(iesn.Ntot, 1) * 2 - 1);

		if(iesn.orthonormalWeights)
			Wr=[];
        		for i=1:seqDim ; 
				wwtemp = rand(Nr(i),Nr(i)); wwtemp = iesn.rho * orth(wwtemp);
				Wr = blkdiag(Wr,wwtemp);
			end
		else
        		Wr = full(sprand(iesn.Ntot-size(iesn.Wr,1),iesn.Ntot-size(iesn.Wr,1), iesn.connectivity));
		        Wr(Wr ~= 0) = Wr(Wr ~= 0) * 2 - 1;
        		Wr = Wr * (iesn.rho / max(abs(eig(Wr))));
			Wr = blkdiag(zeros(size(iesn.Wr,1),size(iesn.Wr,1)),Wr);
		end
		iesn.Wr = blkdiag(iesn.Wr , Wr(1+size(iesn.Wr,1):end,1+size(iesn.Wr,1):end));
        end
        function train(iesn, trX, trY, washout,varargin)
        % Trains the network on input X given target Y.
        %
        % args: 
        %   trX: cell array of size N x 1 time series. Each cell contains an
        %   array of size sequenceLenght x sequenceDimension.
        %   trY: target matrix composed by all sequences. Washout must be 
        %   applied before calling this function.
        %   washout: number of initial timesteps not to collect.
        
        if( iesn.seqDim ~= size(trX{1},2))
                    error('seqDim mismatch');
	end
      
	N = length(trX);
        trainLen = size(trY,1);

  	numvarargs = length(varargin);
  	for i = 1:2:numvarargs
  	    switch varargin{i}
  	        case 'XX', X = varargin{i+1};
  	        otherwise, error('the option does not exist');
  	    end
  	end
        
	if exist('X','var') ~= 1
	        X = zeros(1+iesn.seqDim+iesn.Ntot, trainLen);
	        idx = 1;
	        for s = 1:N
	            U = trX{s}';
	            x = zeros(iesn.Ntot,1);
	            for i = 1:size(U,2)
	                u = U(:,i);
	                x_ = feval(iesn.nonlinearfunction,iesn.Win*u + iesn.Wr*x + iesn.Wb); 
	                x = (1-iesn.alpha)*x + iesn.alpha*x_;
	                if i > washout
	                    X(:,idx) = [1;u;x];
	                    idx = idx+1;
	                end
	            end
	        end
	end

        iesn.internalState = X(1+iesn.seqDim+1:end,:);
        iesn.Wout = feval(iesn.readout_training, X, trY, iesn);

        end
        function y = predict(iesn, data, washout)
        % Computes the output given the data.
        %
        % args:
        %   data: cell array of size N x 1 time series. Each cell contains an
        %   array of size sequenceLenght x sequenceDimension.
        %   washout: number of initial timesteps to not collect.
        %
        % returns:
        %   y: predicted output.
            
            iesn.seqDim = size(data{1},2);
            N = length(data);
            trainLen = 0;
            for s = 1:N
                trainLen = trainLen + size(data{s},1) - washout;
            end
            
            X = zeros(1+iesn.seqDim+iesn.Ntot, trainLen);
            idx = 1;
            for s = 1:N
                U = data{s}';
                x = zeros(iesn.Ntot,1);
                
                for i = 1:size(U,2)
                    u = U(:,i);
                    x_ = feval(iesn.nonlinearfunction,iesn.Win*u + iesn.Wr*x + iesn.Wb); 
                    x = (1-iesn.alpha)*x + iesn.alpha*x_;
                    if i > washout
                        X(:,idx) = [1;u;x];
                        idx = idx+1;
                    end
                end
            end
            
            iesn.internalState = X(1+iesn.seqDim+1:end,:);
            y = iesn.Wout*X;
            y = y';
        end
    end
end
