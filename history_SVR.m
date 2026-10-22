
%% HISTORY-BASED SVR: DATA PREPARATION AND BASELINE
% MATLAB R2023a
% Requires a completed Simulink simulation saved as "out".
%
% Inputs:
%   Iload, Vpcc
% Target:
%   Iref = reference compensating current
%
% Four features:
%   Iload(k), Iload(k-1), Iload(k-2), Vpcc(k)

clc;
clearvars -except out;

%% 1. Retrieve Simulink output
assert(exist('out','var') == 1, ...
    'Run the Simulink model first. Variable out is missing.');

simOut = out;

%% 2. Extract logged signals
loadStruct = simOut.get('Iload');
refStruct  = simOut.get('Iref');
vPccStruct = simOut.get('Vpcc');

assert(~isempty(loadStruct), 'Logged signal Iload is missing.');
assert(~isempty(refStruct),  'Logged signal Iref is missing.');
assert(~isempty(vPccStruct),'Logged signal Vpcc is missing.');

tLoad = loadStruct.time(:);
Iload = squeeze(loadStruct.signals.values);
Iload = Iload(:);

tRef = refStruct.time(:);
Iref = squeeze(refStruct.signals.values);
Iref = Iref(:);

tV = vPccStruct.time(:);
Vpcc = squeeze(vPccStruct.signals.values);
Vpcc = Vpcc(:);

assert(numel(tLoad) == numel(Iload), ...
    'Iload time/data length mismatch.');
assert(numel(tRef) == numel(Iref), ...
    'Iref time/data length mismatch.');
assert(numel(tV) == numel(Vpcc), ...
    'Vpcc time/data length mismatch.');

%% 3. Synchronize signals
% Use Iload time as the common time base.
t = tLoad;

Iref = interp1(tRef, Iref, t, 'linear', NaN);
Vpcc = interp1(tV, Vpcc, t, 'linear', NaN);

%% 4. Construct four history-based features
% Order MUST match the Simulink Mux and evaluation script.
X = [Iload(3:end), ...
     Iload(2:end-1), ...
     Iload(1:end-2), ...
     Vpcc(3:end)];

% Target: reference compensating current
Y = Iref(3:end);
t = t(3:end);

%% 5. Remove invalid samples
valid = all(isfinite(X),2) & ...
        isfinite(Y) & isfinite(t);

X = X(valid,:);
Y = Y(valid);
t = t(valid);

assert(size(X,1) > 100, ...
    'Too few valid samples. Check logged signals.');

fprintf('\nHISTORY-BASED SVR DATA\n');
fprintf('Valid samples: %d\n', size(X,1));
fprintf('Input features: %d\n', size(X,2));
fprintf('Target range: %.6f to %.6f A\n', min(Y), max(Y));
fprintf('Target RMS: %.6f A\n', sqrt(mean(Y.^2)));

if sqrt(mean(Y.^2)) < 1e-6
    warning(['Iref is nearly zero. Verify that Iref logs the ', ...
        'reference compensating current, not an incorrect signal.']);
end

%% 6. Reduce data for initial training, if necessary
% Chronological subsampling. The feature order is unchanged.
maxSamples = 10000;

if size(X,1) > maxSamples
    idx = unique(round(linspace(1,size(X,1),maxSamples)));
    X = X(idx,:);
    Y = Y(idx);
    t = t(idx);
end

fprintf('Samples retained: %d\n', size(X,1));

%% 7. Chronological training/test split
N = size(X,1);
nTrain = floor(0.80*N);

assert(nTrain > 10 && nTrain < N, ...
    'Insufficient samples for training and testing.');

XTrain = X(1:nTrain,:);
YTrain = Y(1:nTrain);

XTest = X(nTrain+1:end,:);
YTest = Y(nTrain+1:end);
tTest = t(nTrain+1:end);

%% 8. Train baseline SVR
fprintf('\nTraining baseline history-based SVR...\n');

svrHistory = fitrsvm(XTrain,YTrain, ...
    'KernelFunction','gaussian', ...
    'Standardize',true);

%% 9. Evaluate baseline SVR
Ypred = predict(svrHistory,XTest);

err = YTest-Ypred;

RMSE = sqrt(mean(err.^2));
MAE = mean(abs(err));
targetRMS = sqrt(mean(YTest.^2));
NRMSE = 100*RMSE/max(targetRMS,eps);

fprintf('\nBASELINE HISTORY-BASED SVR\n');
fprintf('Test samples: %d\n',numel(YTest));
fprintf('RMSE: %.6f A\n',RMSE);
fprintf('MAE: %.6f A\n',MAE);
fprintf('NRMSE: %.2f %% of target RMS\n',NRMSE);

%% 10. Plot baseline prediction
nShow = min(1000,numel(YTest));

figure;
plot(tTest(1:nShow),YTest(1:nShow),'b','LineWidth',1.2);
hold on;
plot(tTest(1:nShow),Ypred(1:nShow),'r--','LineWidth',1.2);
xlabel('Time (s)');
ylabel('Current (A)');
legend('Target Iref','Baseline SVR prediction');
title('Baseline history-based SVR');
grid on;

figure;
plot(tTest(1:nShow),err(1:nShow),'LineWidth',1.2);
xlabel('Time (s)');
ylabel('Prediction error (A)');
title('Baseline SVR prediction error');
grid on;

%% 11. Save the dataset for tuning and evaluation
featureNames = {'Iload_k','Iload_kminus1', ...
                'Iload_kminus2','Vpcc_k'};
targetName = 'Iref_reference_compensating_current';

save('SVR_AHF_History_Data.mat', ...
    'X','Y','t','nTrain','featureNames','targetName');

%% 12. Save baseline model and metrics
save('SVR_AHF_History_Model.mat', ...
    'svrHistory','RMSE','MAE','NRMSE');

fprintf('\nSaved dataset: SVR_AHF_History_Data.mat\n');
fprintf('Saved baseline model: SVR_AHF_History_Model.mat\n');
