function [Wr] = EchoStateGC_reference_Wr_build(Nr,nodes,varargin)

	rho=0.9;	% default value
	numvarargs = length(varargin);
	for i = 1:2:numvarargs
        	switch varargin{i}
        	        case 'spectralRadius', rho = varargin{i+1};
        	        case 'rho', rho = varargin{i+1};
        	        otherwise, error('the option does not exist');
        	end
        end
        
	rhoCent=round(rho*10000);

	folder = [ fileparts(mfilename('fullpath')) sprintf('/reference_Wr/rho_%04d',rhoCent) ]	

	if ~exist(folder, 'dir')
		mkdir(folder);
	end

	folder = [folder sprintf('/nodes_%d',nodes)];
	if ~exist(folder, 'dir')
		mkdir(folder);
	end
	
	matrixfile=[folder sprintf('/Wr_N_%04d.mat',Nr)];
	
	if ~exist(matrixfile, 'file')	
		Wr=[];
		for i=1:nodes ; 
			wwtemp = rand(Nr,Nr); wwtemp = rho * orth(wwtemp);
			Wr = blkdiag(Wr,wwtemp);
		end
		save(matrixfile,'Wr');
	else
		load(matrixfile);
	end
end
