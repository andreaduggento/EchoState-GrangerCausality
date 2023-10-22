function  [ gc, lambda ] = EchoStateGC_GCx(dataX, Nr, leakRate, spectralRadius, ...
    inputScaling, reg, readout_training, nonlinearfunction, washout)

arguments
    dataX
    Nr double = 25
    leakRate double = 0.6
    spectralRadius double = 0.9
    inputScaling double = 0
    reg double = 1e-8
    readout_training string = 'elasticridgeregression'
    nonlinearfunction string = 'tanh'  % tanh, relu, relog, logan, iden
    washout int32 = 0  % recommended: 0.005*len
end

numSes = size(dataX,1);
[len, nodes]= size(dataX{1});

Wr=[];
for i=1:nodes
	wwtemp = rand(Nr,Nr); 
    wwtemp = spectralRadius * orth(wwtemp);
	Wr = blkdiag(Wr,wwtemp);
end
Win = []; 
for i=1:nodes
	wtemp = inputScaling*(rand(Nr,1) * 2 - 1); 
	Win=blkdiag(Win,wtemp); 
end

errors=zeros(nodes,1);

trY=sparse(numSes*(len-1-washout),nodes);
trYY=sparse(numSes*(len-1-washout),nodes-1);
trX=cell(numSes,1);
trXY=cell(numSes,1);
for ts = 1 : numSes
	trX{ts} = zeros(len-1,nodes);
	trXY{ts} = zeros(len-1,nodes-1);
end

for ts = 1 : numSes
	trX{ts} = dataX{ts}(1:end-1,:);
end

%%%%%%%   UNRESTRICTED MODELS
for ts = 1 : numSes
	trY((ts-1)*(len-1-washout)+1:ts*(len-1-washout),:) = dataX{ts}(2+washout:end,:);
end
esnAll = EchoStateGC_iESN(Nr*ones(1,nodes),nodes,'Wr',Wr,'Win',Win,'leakRate', ...
    leakRate,'spectralRadius',spectralRadius,'regularization',reg, ...
    'nonlinearfunction',nonlinearfunction,'readoutTraining',readout_training);
esnAll.train(trX,trY,washout);
output = esnAll.predict(trX,washout);
for jj=1:nodes  
    errors(jj)= mean_squared_error(output(1:end,jj), trY(1:end,jj)); 
end

%%%%%%%   RESTRICTED MODELS
errors2=zeros(nodes,nodes);
gc=zeros(nodes,nodes);
esn = cell(nodes, 1);
for j = 1 : nodes
	for ts = 1 : numSes
		trXY{ts} = dataX{ts}(1:end-1,(1:nodes) ~= j);
	end
 	trYY = trY(1:end,(1:nodes) ~= j);
	wtemp=esnAll.Wr(kron(1:nodes,ones(1,Nr))~=j,kron(1:nodes,ones(1,Nr))~=j);
	wintemp=Win(kron(1:nodes,ones(1,Nr))~=j,(1:nodes) ~= j);
	esn{j} = EchoStateGC_iESN(Nr*ones(1,nodes-1),nodes-1,'Wr',wtemp, ...
        'Win',wintemp,'leakRate',leakRate,'spectralRadius',spectralRadius, ...
        'regularization',esnAll.lambda,'nonlinearfunction',nonlinearfunction);
	esn{j}.train(trXY,trYY,washout);
	output2 = esn{j}.predict(trXY,washout);
	for i = 1:nodes
		if (i==j)
			gc(j,i)=0.;
			errors2(i,j)= 0.;   % pleonastico
		elseif (i<j) 
			errors2(i,j)= mean_squared_error(output2(1:end,i),trYY(1:end,i));
			gc(j,i) = log(errors2(i,j)/errors(i));
		else	
			errors2(i,j)= mean_squared_error(output2(1:end,i-1),trYY(1:end,i-1));
			gc(j,i) = log(errors2(i,j)/errors(i));
		end
    end
	lambda = esnAll.lambda;
end


