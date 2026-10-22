
%% INSPECT CONVENTIONAL AHF OUTPUT
clc;

assert(evalin('base', "exist('out','var')") == 1, ...
    'Run the Simulink simulation first.');

out = evalin('base', 'out');

disp('Output object class:');
disp(class(out));

disp('Available output names:');

if isa(out, 'Simulink.SimulationOutput')
    disp(out.who);
else
    disp(fieldnames(out));
end

disp('Checking expected signals...');

names = {'Isource','Iref','Iload','Vpcc','Vdc'};

for k = 1:numel(names)
    try
        sig = out.get(names{k});
        fprintf('\n%s: found\n', names{k});
        disp(class(sig));

        if isa(sig, 'timeseries')
            fprintf('Number of time samples: %d\n', ...
                numel(sig.Time));
            fprintf('Data dimensions: ');
            disp(size(sig.Data));
        end
    catch
        fprintf('\n%s: not found by this name\n', ...
            names{k});
    end
end
