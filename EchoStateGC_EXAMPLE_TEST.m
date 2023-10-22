%% Load example data
load 'example_data_Nnet_169_W_020_N_10k__tk5.mat'; 

exp_normed = normalize(Expression1, 1);
[len,nodes] = size(exp_normed);

data = cell(1, 1);
data{1} = exp_normed;

GC = EchoStateGC_GCx(data, 25);

GC
