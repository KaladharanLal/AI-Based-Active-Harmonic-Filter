
%% SINGLE-PHASE AHF - SVR TRAINING
% Uses the existing Simulink SimulationOutput object named "out"

clc;

%% 1. Get the simulation output
% Do NOT use clear or clearvars here.
% They would remove the variable "out".

assert(evalin('base',"exist('out','var')") == 1, ...
    'Variable out is missing. Run the Simulink model first.');

out = evalin('base','out');

%% 2. Extract logged signals
% Your screenshot confirms these signals exist inside out.

loadStruct = out.get('Iload');
refStruct  = out.get('Iref');
vPccStruct = out.get('Vpcc');

% Structure With Time uses lowercase fields: time, signals
tLoad = loadStruct.time(:);
Iload = squeeze(loadStruct.signals.values);
Iload = Iload(:);

tRef = refStruct.time(:);
Iref = squeeze(refStruct.signals.values);
Iref = Iref(:);

tV = vPccStruct.time(:);
Vpcc = squeeze(vPccStruct.signals.values);
Vpcc = Vpcc(:);

%% 3. Validate and synchronize signals
assert(numel(tLoad) == numel(Iload), ...
    'Load-current time and data lengths differ.');

assert(numel(tRef) == numel(Iref), ...
    'Reference-current time and data lengths differ.');

assert(numel(tV) == numel(Vpcc), ...
    'PCC-voltage time and data lengths differ.');

% Use the load-current time vector as the common time base
t = tLoad;

Iref = interp1(tRef,Iref,t,'linear',NaN);
Vpcc = interp1(tV,Vpcc,t,'linear',NaN);

%% 4. Create SVR input features and target
% Initial feature set:
%   Feature 1 = load current
%   Feature 2 = PCC voltage
%
% Target = verified compensation-current reference
% Confirm that Iref is the desired reference, not the PI error.

X = [Iload,Vpcc];
Y = Iref;

valid = isfinite(t) & all(isfinite(X),2) & isfinite(Y);

t = t(valid);
X = X(valid,:);
Y = Y(valid);

assert(size(X,1) >= 100, ...
    'Too few valid samples. Check the logged data.');

fprintf('Number of valid samples: %d\n',size(X,1));

%% 5. Split data for an initial prediction check
rng(42);

cv = cvpartition(size(X,1),'HoldOut',0.20);

XTrain = X(training(cv),:);
YTrain = Y(training(cv),:);

XTest = X(test(cv),:);
YTest = Y(test(cv),:);

%% 6. Train single-output SVR
svrModel = fitrsvm(XTrain,YTrain, ...
    'KernelFunction','gaussian', ...
    'Standardize',true);

%% 7. Predict test samples
Ypred = predict(svrModel,XTest);

%% 8. Calculate regression errors
RMSE = sqrt(mean((YTest-Ypred).^2));
MAE = mean(abs(YTest-Ypred));

fprintf('\nSVR RESULTS\n');
fprintf('RMSE = %.6f A\n',RMSE);
fprintf('MAE  = %.6f A\n',MAE);

%% 9. Plot target and predicted current
figure;
plot(YTest,'LineWidth',1.1);
hold on;
plot(Ypred,'--','LineWidth',1.1);
xlabel('Test sample index');
ylabel('Current (A)');
legend('Target Iref','SVR predicted Iref');
title('Single-phase AHF: SVR prediction');
grid on;

%% 10. Plot prediction error
figure;
plot(YTest-Ypred);
xlabel('Test sample index');
ylabel('Error (A)');
title('SVR prediction error');
grid on;

%% 11. Save model and training data
save('SVR_AHF_Model.mat','svrModel','RMSE','MAE');
save('AHF_SVR_training.mat','X','Y','t');

fprintf('\nSVR training finished.\n');
fprintf('Saved: SVR_AHF_Model.mat\n');
fprintf('Saved: AHF_SVR_training.mat\n');
