close all;
input = readmatrix('NN_input.txt')';
target = readmatrix('NN_target.txt');

net = fitnet([10 9]);
net = configure(net, input, target);

net.layers{1}.transferFcn = 'tansig';

a = sim(net, input);
net = train(net, input, target);
%t = net(input);
%plotregression(target, t);