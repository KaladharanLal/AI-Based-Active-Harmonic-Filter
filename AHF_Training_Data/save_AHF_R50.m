
%% SAVE CONVENTIONAL AHF DATASET - 50 OHM
clc;

% Obtain the completed Simulink simulation output
assert(evalin('base',"exist('out','var')") == 1, ...
    'Run the conventional AHF simulation first.');

out = evalin('base','out');

% Extract logged structures
Sload = out.get('Iload');
Sref  = out.get('Iref');
Spcc  = out.get('Vpcc');
Sdc   = out.get('Vdc');

% Extract time vectors
tLoad = Sload.time(:);
tRef  = Sref.time(:);
tPcc  = Spcc.time(:);
tDc   = Sdc.time(:);

% Extract measured data
Iload = squeeze(Sload.signals.values);
Iref  = squeeze(Sref.signals.values);
Vpcc  = squeeze(Spcc.signals.values);
Vdc   = squeeze(Sdc.signals.values);

% Ensure column-vector format
Iload = Iload(:);
Iref  = Iref(:);
Vpcc  = Vpcc(:);
Vdc   = Vdc(:);

% Check sample counts
assert(numel(tLoad) == numel(Iload), ...
    'Iload time and data lengths differ.');

assert(numel(tRef) == numel(Iref), ...
    'Iref time and data lengths differ.');

assert(numel(tPcc) == numel(Vpcc), ...
    'Vpcc time and data lengths differ.');

assert(numel(tDc) == numel(Vdc), ...
    'Vdc time and data lengths differ.');

% Check time alignment
assert(isequal(tLoad,tRef) && ...
       isequal(tLoad,tPcc) && ...
       isequal(tLoad,tDc), ...
       'Signals have different time vectors.');

% Construct input and target matrices
% Use these inputs only if they match your ANN design.
X = [Iload, Vpcc, Vdc];
Y = Iref;

time = tLoad;

% Verify data quality
assert(all(isfinite(X(:))) && all(isfinite(Y(:))), ...
    'NaN or Inf values detected.');

% Load parameters for the first simulation
R_load = 50;
L_load = 100e-3;

% Save dataset
save('AHF_Data_R50.mat', ...
    'X','Y','time','R_load','L_load');

% Display results
fprintf('Dataset saved successfully.\n');
fprintf('Samples: %d\n',size(X,1));
fprintf('Input features: %d\n',size(X,2));
fprintf('Target outputs: %d\n',size(Y,2));
