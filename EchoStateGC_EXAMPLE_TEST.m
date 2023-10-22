%% Load example data
load 'example_data_Nnet_169_W_020_N_10k__tk5.mat'; 
data=Expression1;
data=normalize(data,1); 

nodes= size(data,2);
len  = size(data,1);
	

%% Echo-State-Network Parameters	
washout=100;
leakRate=.6;
spectralRadius=.9;
reg=1e-8;
Nr=25;

%% Echo-State-Network nonlinear functions on each neuron 
%% Uncomment as appropiate
fun='tanh';
%fun='relu';
%fun='relog';
%fun='logan';
%fun='iden'; % would make it linear


%% Build Wr matrix, if not already available		
Wr=EchoStateGC_reference_Wr_build(Nr,nodes);

%% Calculate Granger Causality	
GC = EchoStateGC_GCx(data, Nr,'Wr',Wr,'nonlinearfunction',fun);

GC
