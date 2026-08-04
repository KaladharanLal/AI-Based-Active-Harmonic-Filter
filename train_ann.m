clear;
clc;

load ANN_Dataset.mat

net = fitnet(20);      % 20 hidden neurons

net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

[net,tr] = train(net, inputs, targets);

outputs = net(inputs);

figure;
plotperform(tr);

figure;
plotregression(targets, outputs);

figure;
plot(targets(1:2000),'b');
hold on;
plot(outputs(1:2000),'r');
legend('Target','ANN Output');
grid on;

save trainedANN.mat net