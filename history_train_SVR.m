
%% HISTORY-BASED SVR HYPERPARAMETER TUNING
% MATLAB R2023a
% Requires SVR_AHF_History_Data.mat from history_SVR.m

clc;

%% 1. Load the latest dataset
assert(isfile('SVR_AHF_History_Data.mat'), ...
    'Run history_SVR.m first. Dataset file is missing.');

D = load('SVR_AHF_History_Data.mat', ...
    'X','Y','t','nTrain','featureNames','targetName');

X = D.X;
Y = D.Y;
t = D.t;
nTrain = D.nTrain;
featureNames = D.featureNames;
targetName = D.targetName;

assert(size(X,2) == 4, ...
    'Training data must contain four input features.');

assert(size(X,1) == numel(Y), ...
    'X and Y have different sample counts.');

assert(nTrain > 10 && nTrain < size(X,1), ...
    'Invalid chronological training split.');

fprintf('\nSVR HYPERPARAMETER TUNING\n');
fprintf('Samples: %d\n',size(X,1));
fprintf('Features: %d\n',size(X,2));
fprintf('Training samples: %d\n',nTrain);
fprintf('Test samples: %d\n',size(X,1)-nTrain);
fprintf('Target: %s\n',targetName);

%% 2. Chronological split
XTrain = X(1:nTrain,:);
YTrain = Y(1:nTrain);

XTest = X(nTrain+1:end,:);
YTest = Y(nTrain+1:end);

%% 3. Tune SVR hyperparameters
% Optimization is performed using training data only.
rng(42);

fprintf('\nStarting hyperparameter optimization...\n');

svrTuned = fitrsvm(XTrain,YTrain, ...
    'KernelFunction','gaussian', ...
    'Standardize',true, ...
    'OptimizeHyperparameters', ...
        {'BoxConstraint','Epsilon','KernelScale'}, ...
    'HyperparameterOptimizationOptions', ...
        struct( ...
            'KFold',3, ...
            'MaxObjectiveEvaluations',15, ...
            'ShowPlots',false, ...
            'Verbose',1));

fprintf('\nHyperparameter optimization completed.\n');

%% 4. Evaluate on the held-out chronological test set
Ypred = predict(svrTuned,XTest);

err = YTest-Ypred;

RMSE = sqrt(mean(err.^2));
MAE = mean(abs(err));
targetRMS = sqrt(mean(YTest.^2));
NRMSE = 100*RMSE/max(targetRMS,eps);

fprintf('\nTUNED SVR RESULTS\n');
fprintf('Test samples: %d\n',numel(YTest));
fprintf('RMSE: %.6f A\n',RMSE);
fprintf('MAE: %.6f A\n',MAE);
fprintf('NRMSE: %.2f %% of target RMS\n',NRMSE);

%% 5. Display tuned hyperparameters
disp('Tuned hyperparameters:');
disp(svrTuned.ModelParameters);

%% 6. Plot tuned-model prediction
nShow = min(1000,numel(YTest));

figure;
plot(YTest(1:nShow),'b','LineWidth',1.2);
hold on;
plot(Ypred(1:nShow),'r--','LineWidth',1.2);
xlabel('Test sample');
ylabel('Current (A)');
legend('Target Iref','Tuned SVR prediction');
title('Tuned history-based SVR');
grid on;

figure;
plot(err(1:nShow),'LineWidth',1.2);
xlabel('Test sample');
ylabel('Prediction error (A)');
title('Tuned SVR prediction error');
grid on;

%% 7. Save the tuned model and metrics
save('SVR_AHF_Tuned_Model.mat', ...
    'svrTuned','RMSE','MAE','NRMSE', ...
    'nTrain','featureNames','targetName');

fprintf('\nSaved tuned model: SVR_AHF_Tuned_Model.mat\n');
