%% Train a combined GB-RBM + PINN substation digital twin
% Run this example from a folder containing datalogsheet.xlsx, or replace
% dataFile with the full path to your telemetry workbook.

dataFile = fullfile(pwd, 'datalogsheet.xlsx');
if ~isfile(dataFile)
    error('Example data file not found. Set dataFile to the full path of datalogsheet.xlsx.');
end

result = PhyGESS.trainGBRBMPINN(dataFile, ...
    'InputColumn', 2, ...
    'TargetColumn', 4, ...
    'NumSyntheticSamples', 5000, ...
    'GBRBMEpochs', 300, ...
    'PINNEpochs', 300, ...
    'SavePath', fullfile(pwd, 'Combined_GBRBM_PINN_Result.mat'));

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
