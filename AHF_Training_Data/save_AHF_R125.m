
%% SAVE CONVENTIONAL AHF DATASET - 75 OHM
clc;

assert(evalin('base',"exist('out','var')") == 1, ...
    'Run the 75-ohm simulation first.');

out = evalin('base','out');

Sload = out.get('Iload');
Sref  = out.get('Iref');
Spcc  = out.get('Vpcc');
Sdc   = out.get('Vdc');

time = Sload.time(:);

Iload = squeeze(Sload.signals.values);
Iref  = squeeze(Sref.signals.values);
Vpcc  = squeeze(Spcc.signals.values);
Vdc   = squeeze(Sdc.signals.values);

Iload = Iload(:);
Iref  = Iref(:);
Vpcc  = Vpcc(:);
Vdc   = Vdc(:);

assert(numel(time) == numel(Iload) && ...
       numel(time) == numel(Iref) && ...
       numel(time) == numel(Vpcc) && ...
       numel(time) == numel(Vdc), ...
       'Signal sample counts differ.');

assert(isequal(time,Sref.time(:)) && ...
       isequal(time,Spcc.time(:)) && ...
       isequal(time,Sdc.time(:)), ...
       'Signal time vectors differ.');

X = [Iload, Vpcc, Vdc];
Y = Iref;

assert(all(isfinite(X(:))) && all(isfinite(Y(:))), ...
    'NaN or Inf values detected.');

R_load = 125;
L_load = 100e-3;

save('AHF_Data_R125.mat', ...
    'X','Y','time','R_load','L_load');

fprintf('Saved AHF_Data_R125.mat\n');
fprintf('Samples: %d\n',size(X,1));
