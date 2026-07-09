%% Train a combined GB-RBM + PINN substation digital twin
% The example expects datalogsheet.xlsx to live at the repository root.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
dataFile = fullfile(repoRoot, 'datalogsheet.xlsx');
if ~isfile(dataFile)
    error('Example data file not found at the repository root. Set dataFile to the full path of datalogsheet.xlsx.');
end

result = PhyGESS.trainGBRBMPINN(dataFile, ...
    'InputColumn', 2, ...
    'TargetColumn', 4, ...
    'NumSyntheticSamples', 5000, ...
    'GBRBMEpochs', 300, ...
    'PINNEpochs', 300, ...
    'SavePath', fullfile(repoRoot, 'Combined_GBRBM_PINN_Result.mat'));

disp(result.Validation);

figure('Name', 'GB-RBM + PINN Validation', 'Position', [100, 100, 1000, 450]);
plot(result.Validation.Targets, 'k-', 'LineWidth', 1.8);
hold on;
plot(result.Validation.Predictions, 'b--', 'LineWidth', 1.5);
grid on;
xlabel('Sample');
ylabel('Target');
title(sprintf('PINN Validation on Real Telemetry (RMSE %.3f)', result.Validation.RMSE));
legend('Observed', 'PINN prediction', 'Location', 'best');
