
%% SVR diagnostic and improved initial training
clear;
clc;
load('AHF_SVR_training.mat','X','Y','t');

%% 1. Uniformly reduce data for faster training
maxSamples = 10000;

if numel(Y) > maxSamples
    idx = round(linspace(1,numel(Y),maxSamples));
    X = X(idx,:);
    Y = Y(idx);
    t = t(idx);
end

%% 2. Chronological split: first 80% train, last 20% test
N = size(X,1);
nTrain = floor(0.80*N);

XTrain = X(1:nTrain,:);
YTrain = Y(1:nTrain);

XTest = X(nTrain+1:end,:);
YTest = Y(nTrain+1:end);
tTest = t(nTrain+1:end);

%% 3. Train SVR
svrModel = fitrsvm(XTrain,YTrain, ...
    'KernelFunction','gaussian', ...
    'Standardize',true);

%% 4. Predict unseen later samples
Ypred = predict(svrModel,XTest);

%% 5. Evaluate
err = YTest-Ypred;

RMSE = sqrt(mean(err.^2));
MAE = mean(abs(err));
targetRMS = sqrt(mean(YTest.^2));
targetStd = std(YTest);
NRMSE = 100*RMSE/max(targetRMS,eps);

fprintf('Test samples: %d\n',numel(YTest));
fprintf('Target RMS: %.4f A\n',targetRMS);
fprintf('Target standard deviation: %.4f A\n',targetStd);
fprintf('RMSE: %.4f A\n',RMSE);
fprintf('MAE: %.4f A\n',MAE);
fprintf('NRMSE relative to target RMS: %.2f %%\n',NRMSE);

%% 6. Compare waveforms over a short section
nShow = min(1000,numel(YTest));

figure;
plot(tTest(1:nShow),YTest(1:nShow),'LineWidth',1.2);
hold on;
plot(tTest(1:nShow),Ypred(1:nShow),'--','LineWidth',1.2);
xlabel('Time (s)');
ylabel('Current (A)');
legend('Target Iref','SVR prediction');
title('SVR: chronological test segment');
grid on;

%% 7. Error waveform
figure;
plot(tTest(1:nShow),err(1:nShow));
xlabel('Time (s)');
ylabel('Prediction error (A)');
title('SVR error on unseen time segment');
grid on;

%% 8. Save this diagnostic model separately
save('SVR_AHF_Chronological.mat', ...
    'svrModel','RMSE','MAE','NRMSE');
