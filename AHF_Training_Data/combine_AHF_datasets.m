
%% COMBINE FIVE AHF DATASETS
clc;

files = {
    'AHF_Data_R50.mat'
    'AHF_Data_R75.mat'
    'AHF_Data_R100.mat'
    'AHF_Data_R125.mat'
    'AHF_Data_R150.mat'
};

X_all = [];
Y_all = [];
R_all = [];
time_all = [];

for k = 1:numel(files)

    D = load(files{k}, 'X','Y','R_load','time');

    if k == 1
        nInputs = size(D.X,2);
        nTargets = size(D.Y,2);
    end

    assert(size(D.X,2) == nInputs, ...
        'Input feature counts differ.');

    assert(size(D.Y,2) == nTargets, ...
        'Target feature counts differ.');

    assert(size(D.X,1) == size(D.Y,1), ...
        'Input and target row counts differ.');

    X_all = [X_all; D.X];
    Y_all = [Y_all; D.Y];

    R_all = [R_all; ...
        repmat(D.R_load,size(D.X,1),1)];

    time_all = [time_all; D.time(:)];
end

save('AHF_Combined_Dataset.mat', ...
    'X_all','Y_all','R_all','time_all');

fprintf('\nCombined dataset saved.\n');
fprintf('Total samples: %d\n', size(X_all,1));
fprintf('Input features: %d\n', size(X_all,2));
fprintf('Target outputs: %d\n', size(Y_all,2));

disp('Resistance values included:');
disp(unique(R_all)');
