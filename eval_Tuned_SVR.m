
%% EVALUATE TUNED HISTORY-BASED SVR
% MATLAB R2023a
% Requires:
%   SVR_AHF_History_Data.mat
%   SVR_AHF_Tuned_Model.mat

clc;

%% 1. Check required files
assert(isfile('SVR_AHF_History_Data.mat'), ...
    'Dataset missing. Run history_SVR.m first.');

assert(isfile('SVR_AHF_Tuned_Model.mat'), ...
    'Tuned model missing. Run history_train_SVR.m first.');

%% 2. Load the exact dataset and tuned model
D = load('SVR_AHF_History_Data.mat', ...
    'X','Y','t','nTrain','featureNames','targetName');

M = load('SVR_AHF_Tuned_Model.mat', ...
    'svrTuned','nTrain','featureNames','targetName');

X = D.X;
Y = D.Y;
t = D.t;

svrTuned = M.svrTuned;

assert(size(X,2) == 4, ...
    'Expected four input features.');

assert(size(X,1) == numel(Y), ...
    'Input-target sample count mismatch.');

assert(numel(t) == numel(Y), ...
    'Time-target sample count mismatch.');

assert(D.nTrain == M.nTrain, ...
    'Dataset and model use different training splits.');

assert(isequal(D.featureNames,M.featureNames), ...
    'Dataset and model feature definitions differ.');

assert(strcmp(D.targetName,M.targetName), ...
    'Dataset and model target definitions differ.');

nTrain = D.nTrain;

%% 3. Reproduce the chronological test split
XTest = X(nTrain+1:end,:);
YTest = Y(nTrain+1:end);
tTest = t(nTrain+1:end);

assert(~isempty(YTest), ...
    'No test samples available.');

fprintf('\nTUNED SVR EVALUATION\n');
fprintf('Test samples: %d\n',numel(YTest));
fprintf('Target range: %.6f to %.6f A\n', ...
    min(YTest),max(YTest));
fprintf('Target RMS: %.6f A\n',sqrt(mean(YTest.^2)));

%% 4. Predict reference compensating current
Ypred = predict(svrTuned,XTest);

%% 5. Calculate evaluation metrics
err = YTest-Ypred;

RMSE = sqrt(mean(err.^2));
MAE = mean(abs(err));
targetRMS = sqrt(mean(YTest.^2));
NRMSE = 100*RMSE/max(targetRMS,eps);

fprintf('\nTest RMSE: %.6f A\n',RMSE);
fprintf('Test MAE: %.6f A\n',MAE);
fprintf('Test NRMSE: %.2f %% of target RMS\n',NRMSE);

%% 6. Plot actual and predicted current
nShow = min(1000,numel(YTest));

figure;
plot(tTest(1:nShow),YTest(1:nShow), ...
    'b','LineWidth',1.2);
hold on;
plot(tTest(1:nShow),Ypred(1:nShow), ...
    'r--','LineWidth',1.2);

xlabel('Time (s)');
ylabel('Current (A)');
legend('Actual reference current', ...
       'Tuned SVR prediction','Location','best');
title('Tuned SVR: Chronological Test');
grid on;

%% 7. Plot prediction error
figure;
plot(tTest(1:nShow),err(1:nShow),'LineWidth',1.2);

xlabel('Time (s)');
ylabel('Prediction error (A)');
title('Tuned SVR Prediction Error');
grid on;

%% 8. Save evaluation results separately
save('SVR_AHF_Tuned_Evaluation.mat', ...
    'RMSE','MAE','NRMSE','YTest','Ypred','err','tTest');

fprintf('\nSaved evaluation: SVR_AHF_Tuned_Evaluation.mat\n');
