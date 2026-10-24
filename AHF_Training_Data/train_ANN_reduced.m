
%% TRAIN ANN USING REDUCED MULTI-LOAD DATASET
clc;
close all;

%% 1. Verify reduced data and indices
assert(size(Xs,1) == size(Ys,1) && ...
       size(Xs,1) == numel(Rs), ...
       'Reduced data dimensions do not match.');

assert(max([trainInd_small; valInd_small; ...
    testInd_small]) <= size(Xs,1), ...
    'Split indices exceed the reduced dataset.');

%% 2. Arrange data for MATLAB fitnet
Xsmall = Xs';
Ysmall = Ys';

%% 3. Configure the network
hiddenNeurons = 20;

netSmall = fitnet(hiddenNeurons,'trainscg');

netSmall.divideFcn = 'divideind';
netSmall.divideParam.trainInd = trainInd_small';
netSmall.divideParam.valInd = valInd_small';
netSmall.divideParam.testInd = testInd_small';

netSmall.trainParam.epochs = 100;
netSmall.trainParam.max_fail = 6;
netSmall.trainParam.showWindow = false;

%% 4. Train
[netSmall,trSmall] = train(netSmall,Xsmall,Ysmall);

%% 5. Predict held-out test samples only
Ytest = Ysmall(:,testInd_small);
YpredTest = netSmall(Xsmall(:,testInd_small));

%% 6. Calculate test performance
errTest = Ytest(:) - YpredTest(:);

rmseTest = sqrt(mean(errTest.^2));
maeTest = mean(abs(errTest));

fprintf('\nREDUCED ANN TEST PERFORMANCE\n');
fprintf('Test RMSE: %.6f A\n',rmseTest);
fprintf('Test MAE : %.6f A\n',maeTest);

%% 7. Calculate performance for each load
Rtest = Rs(testInd_small);

fprintf('\nPer-load test performance:\n');
fprintf('Resistance | RMSE (A) | MAE (A)\n');

resistances = [50 75 100 125 150];

for k = 1:numel(resistances)

    mask = Rtest == resistances(k);
    e = Ytest(mask) - YpredTest(mask);

    fprintf('%9g | %8.4f | %7.4f\n', ...
        resistances(k), ...
        sqrt(mean(e.^2)), ...
        mean(abs(e)));
end

%% 8. Save model and evaluation data
sampleStep = 20;

save('ANN_AHF_MultiLoad_Reduced.mat', ...
    'netSmall','trSmall','hiddenNeurons', ...
    'sampleStep','trainInd_small', ...
    'valInd_small','testInd_small', ...
    'rmseTest','maeTest','-v7.3');

%% 9. Plot a short test segment
nPlot = min(2000,numel(testInd_small));

figure;
plot(Ytest(1:nPlot),'LineWidth',1);
hold on;
plot(YpredTest(1:nPlot),'LineWidth',1);
grid on;

xlabel('Test sample index');
ylabel('Reference current (A)');
legend('Conventional target','ANN prediction');
title('Reduced-data ANN test performance');
