
%% ANN TRAINING - MULTIPLE AHF LOADS
clc;
clear;
close all;

%% 1. Load the combined database
D = load('AHF_Combined_Dataset.mat');

X_all = D.X_all;
Y_all = D.Y_all;
R_all = D.R_all;

assert(size(X_all,1) == size(Y_all,1), ...
    'Input and target sample counts differ.');

assert(all(isfinite(X_all(:))) && ...
       all(isfinite(Y_all(:))), ...
       'Dataset contains NaN or Inf values.');

%% 2. Create chronological train/validation/test indices
resistances = [50 75 100 125 150];

trainInd = [];
valInd = [];
testInd = [];

for k = 1:numel(resistances)

    idx = find(R_all == resistances(k));
    n = numel(idx);

    nTrain = floor(0.70*n);
    nVal   = floor(0.15*n);

    trainInd = [trainInd; idx(1:nTrain)];
    valInd   = [valInd; ...
        idx(nTrain+1:nTrain+nVal)];
    testInd  = [testInd; ...
        idx(nTrain+nVal+1:end)];

end

fprintf('Training samples: %d\n',numel(trainInd));
fprintf('Validation samples: %d\n',numel(valInd));
fprintf('Testing samples: %d\n',numel(testInd));

%% 3. Convert data to neural network format
% MATLAB fitnet expects features and samples in columns.
X = X_all';
Y = Y_all';

%% 4. Create the ANN
hiddenNeurons = 20;

net = fitnet(hiddenNeurons,'trainscg');

% Use explicitly defined chronological partitions
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd';
net.divideParam.valInd   = valInd';
net.divideParam.testInd  = testInd';

% Training parameters
net.trainParam.epochs = 100;
net.trainParam.max_fail = 6;
net.trainParam.showWindow = true;

% fitnet uses input/target normalization by default.
% The network learns normalization parameters from training data.

%% 5. Train the ANN
[net,tr] = train(net,X,Y);

%% 6. Predict the dataset samples
Ypred = net(X);

%% 7. Calculate test-set performance
actualTest = Y(testInd);
predTest   = Ypred(testInd);

rmse = sqrt(mean((actualTest-predTest).^2));
mae  = mean(abs(actualTest-predTest));

fprintf('\nANN TEST PERFORMANCE\n');
fprintf('RMSE: %.6f\n',rmse);
fprintf('MAE : %.6f\n',mae);

%% 8. Save the trained model
save('ANN_AHF_MultiLoad.mat', ...
    'net','tr','hiddenNeurons', ...
    'trainInd','valInd','testInd', ...
    'rmse','mae');

%% 9. Plot test target versus ANN prediction
figure;
plot(actualTest(1:min(2000,numel(testInd))), ...
     'LineWidth',1);
hold on;
plot(predTest(1:min(2000,numel(testInd))), ...
     'LineWidth',1);
grid on;
xlabel('Test sample index');
ylabel('Current (A)');
legend('Conventional target','ANN prediction');
title('ANN prediction on held-out test samples');
