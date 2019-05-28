function  [gc] = EchoStateGC_GC(data, Nr, varargin)

if (size(data,2) > size(data,1))
	data=data';
end

[len,nodes]= size(data);

leakRate=.6;
spectralRadius=.9;
inputScaling=1.;
biasScaling=1.;
reg=1e-8;
washout=round(len*0.005);
nonlinearfunction='tanh';

	numvarargs = length(varargin);
	for i = 1:2:numvarargs
        	switch varargin{i}
        	        case 'spectralRadius', spectralRadius = varargin{i+1};
        	end
        end

Wr=[];
for i=1:nodes ; 
	wwtemp = rand(Nr,Nr); wwtemp = spectralRadius * orth(wwtemp);
	Wr = blkdiag(Wr,wwtemp);
end
	numvarargs = length(varargin);
	for i = 1:2:numvarargs
        	switch varargin{i}
        		case 'leakRate', leakRate = varargin{i+1};
        	        case 'spectralRadius', spectralRadius = varargin{i+1};
        	        case 'inputScaling', inputScaling = varargin{i+1};
        	        case 'biasScaling', biasScaling = varargin{i+1};
        	        case 'regularization', reg = varargin{i+1};
        	        case 'readoutTraining', readout_training = varargin{i+1};
        	        case 'nonlinearfunction', nonlinearfunction = varargin{i+1};
			case 'washout',washout=varargin{i+1};
        	        case 'Wr', Wr = varargin{i+1};
        	        otherwise, error('the option does not exist');
        	end
        end
        

errors=zeros(nodes,1);

%%%%%%%   ALL UNRESTRICTED MODELS
trX{1} = data(1:end-1,[1:end]);
trY = data(2+washout:end,[1:end]);
esnAll = iESN(Nr*ones(1,nodes),nodes,'Wr',Wr,'leakRate',leakRate,'spectralRadius',spectralRadius,'regularization',reg,'nonlinearfunction',nonlinearfunction);
esnAll.train(trX,trY,washout);
output = esnAll.predict(trX,washout);
for jj=1:nodes   errors(jj)= immse(output(1:end,jj), trY(1:end,jj)); end

%%%%%%%   RESTRICTED MODELS
errors2=zeros(nodes,nodes);
gc=zeros(nodes,nodes);

for j = 1 : nodes
	trXY{1} = data(1:end-1,[1:nodes] ~= j);
	trYY = data(2+washout:end,[1:nodes] ~= j);
	wtemp=esnAll.Wr(kron(1:nodes,ones(1,Nr))~=j,kron(1:nodes,ones(1,Nr))~=j);
	esn{j} = iESN(Nr*ones(1,nodes-1),nodes-1,'Wr',wtemp,'leakRate',leakRate,'spectralRadius',spectralRadius,'regularization',reg);
	esn{j}.train(trXY,trYY,washout);
	output2 = esn{j}.predict(trXY,washout);
	for i=1:nodes
		if (i==j)
			gc(j,i)=0.;
			errors2(i,j)= 0.;   % pleonastico
		elseif (i<j) 
			errors2(i,j)= immse(output2(1:end,i),trYY(1:end,i));
			gc(j,i) = log(errors2(i,j)/errors(i));
		else	
			errors2(i,j)= immse(output2(1:end,i-1),trYY(1:end,i-1));
			gc(j,i) = log(errors2(i,j)/errors(i));
		end
	end
end


