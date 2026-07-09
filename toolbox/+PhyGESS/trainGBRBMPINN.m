function result = trainGBRBMPINN(dataSource, varargin)
%TRAINGBRBMPINN Train the combined GB-RBM + PINN digital twin pipeline.
%
%   result = PhyGESS.trainGBRBMPINN(dataSource) reads substation
%   telemetry, splits it into training and holdout sets, trains a
%   Gaussian-Bernoulli RBM on the training split, synthesizes additional
%   telemetry, trains a PINN on the generated data, and validates the PINN
%   against the holdout telemetry.
%
%   dataSource can be a table, timetable, numeric matrix, CSV file, or Excel
%   file. By default, table/file inputs drop the first column as a timestamp.
%   InputColumn and TargetColumn are column numbers after that timestamp
%   column is removed.

if nargin < 1 || isempty(dataSource)
    dataSource = 'datalogsheet.xlsx';
end

p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'FirstColumnIsTime', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'InputColumn', 2, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'TargetColumn', 4, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'NumSyntheticSamples', 10000, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'GBRBMHiddenUnits', 50, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'GBRBMEpochs', 500, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'GBRBMLearningRate', 0.005, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'GBRBMWeightPenalty', 0.001, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'GBRBMMomentum', 0.9, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'GBRBMBurnInSteps', 30, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'HoldoutFraction', 0.2, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'PINNEpochs', 600, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'PINNLearningRate', 0.005, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'PINNLayerWidth', 20, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'PINNHiddenLayers', 2, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'PINNPhysicsWeight', 0.1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'PhysicsK', 0.0002, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'AmbientTemperature', 30, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'ThermalTau', 15, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'UseDerivativePhysics', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'RandomSeed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'Verbose', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'ShowPINNProgress', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'SavePath', '', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});

opts = p.Results;
opts.FirstColumnIsTime = logical(opts.FirstColumnIsTime);
opts.InputColumn = round(opts.InputColumn);
opts.TargetColumn = round(opts.TargetColumn);
opts.NumSyntheticSamples = round(opts.NumSyntheticSamples);
opts.HoldoutFraction = double(opts.HoldoutFraction);
opts.UseDerivativePhysics = logical(opts.UseDerivativePhysics);
opts.Verbose = logical(opts.Verbose);
opts.ShowPINNProgress = logical(opts.ShowPINNProgress);

if ~isempty(opts.RandomSeed)
    oldState = rng(opts.RandomSeed);
    cleanup = onCleanup(@() rng(oldState));
end

[telemetry, sourceInfo] = localReadTelemetry(dataSource, opts.FirstColumnIsTime);
if size(telemetry, 2) < max(opts.InputColumn, opts.TargetColumn)
    error('PhyGESS:ColumnOutOfRange', ...
        'InputColumn and TargetColumn must be within the %d telemetry columns.', size(telemetry, 2));
end

validRows = all(isfinite(telemetry), 2);
if ~all(validRows)
    warning('PhyGESS:RemovedTelemetryRows', 'Removing %d telemetry rows with NaN or Inf values.', nnz(~validRows));
    telemetry = telemetry(validRows, :);
end

if opts.Verbose
    fprintf('Loaded telemetry: %d samples x %d variables.\n', size(telemetry, 1), size(telemetry, 2));
end

numSamples = size(telemetry, 1);
if numSamples < 2
    error('PhyGESS:InsufficientTelemetry', 'At least two telemetry samples are required for a holdout split.');
end

numHoldout = max(1, round(opts.HoldoutFraction * numSamples));
numHoldout = min(numHoldout, numSamples - 1);
splitOrder = randperm(numSamples);
holdoutIdx = splitOrder(1:numHoldout);
trainIdx = splitOrder(numHoldout + 1:end);

trainingTelemetry = telemetry(trainIdx, :);
holdoutTelemetry = telemetry(holdoutIdx, :);

if opts.Verbose
    fprintf('Holdout split: %d training samples, %d holdout samples (%.0f%%).\n', ...
        size(trainingTelemetry, 1), size(holdoutTelemetry, 1), opts.HoldoutFraction * 100);
end

muData = mean(trainingTelemetry, 1);
sigmaData = std(trainingTelemetry, 0, 1);
sigmaData(sigmaData == 0) = 1;
normalizedTelemetry = (trainingTelemetry - muData) ./ sigmaData;

if opts.Verbose
    fprintf('Training GB-RBM and generating %d synthetic samples.\n', opts.NumSyntheticSamples);
end

[rbm, rbmInfo] = PhyGESS.trainGBRBM(normalizedTelemetry', ...
    'NumHiddenUnits', opts.GBRBMHiddenUnits, ...
    'MaxEpochs', opts.GBRBMEpochs, ...
    'LearningRate', opts.GBRBMLearningRate, ...
    'WeightPenalty', opts.GBRBMWeightPenalty, ...
    'Momentum', opts.GBRBMMomentum, ...
    'Verbose', opts.Verbose);

[syntheticTelemetry, syntheticTelemetryNormalized] = PhyGESS.generateGBRBMSamples( ...
    rbm, opts.NumSyntheticSamples, muData, sigmaData, ...
    'BurnInSteps', opts.GBRBMBurnInSteps);

if opts.Verbose
    fprintf('Training PINN on synthetic column %d -> column %d.\n', opts.InputColumn, opts.TargetColumn);
end

[pinnModel, pinnInfo] = PhyGESS.trainPINN( ...
    syntheticTelemetry(:, opts.InputColumn), syntheticTelemetry(:, opts.TargetColumn), ...
    'MaxEpochs', opts.PINNEpochs, ...
    'LearningRate', opts.PINNLearningRate, ...
    'LayerWidth', opts.PINNLayerWidth, ...
    'NumHiddenLayers', opts.PINNHiddenLayers, ...
    'PhysicsWeight', opts.PINNPhysicsWeight, ...
    'PhysicsK', opts.PhysicsK, ...
    'AmbientTemperature', opts.AmbientTemperature, ...
    'ThermalTau', opts.ThermalTau, ...
    'UseDerivativePhysics', opts.UseDerivativePhysics, ...
    'Verbose', opts.Verbose, ...
    'ShowProgress', opts.ShowPINNProgress);

validation = PhyGESS.validatePINN( ...
    pinnModel, holdoutTelemetry(:, opts.InputColumn), holdoutTelemetry(:, opts.TargetColumn));

if opts.Verbose
    fprintf('Holdout validation | RMSE %.4f | MAE %.4f | Bias %.4f\n', ...
        validation.RMSE, validation.MAE, validation.Bias);
end

result = struct();
result.Source = sourceInfo;
result.Options = opts;
result.GBRBM = rbm;
result.GBRBMTrainingInfo = rbmInfo;
result.TelemetryMean = muData;
result.TelemetryStd = sigmaData;
result.RealTelemetry = trainingTelemetry;
result.HoldoutTelemetry = holdoutTelemetry;
result.SyntheticTelemetry = syntheticTelemetry;
result.SyntheticTelemetryNormalized = syntheticTelemetryNormalized;
result.PINN = pinnModel;
result.PINNTrainingInfo = pinnInfo;
result.Validation = validation;

if strlength(string(opts.SavePath)) > 0
    save(opts.SavePath, 'result');
    if opts.Verbose
        fprintf('Saved combined GB-RBM + PINN result to %s\n', string(opts.SavePath));
    end
end
end

function [telemetry, sourceInfo] = localReadTelemetry(dataSource, firstColumnIsTime)
sourceInfo = struct('Type', '', 'Location', '');

if istable(dataSource)
    tableData = dataSource;
    sourceInfo.Type = 'table';
elseif istimetable(dataSource)
    tableData = timetable2table(dataSource, 'ConvertRowTimes', false);
    sourceInfo.Type = 'timetable';
elseif isnumeric(dataSource)
    telemetry = double(dataSource);
    sourceInfo.Type = 'numeric';
    sourceInfo.Location = 'workspace';
    return;
elseif ischar(dataSource) || isstring(dataSource)
    fileName = string(dataSource);
    if ~isfile(fileName)
        error('PhyGESS:FileNotFound', 'Telemetry file not found: %s', fileName);
    end
    tableData = readtable(fileName);
    sourceInfo.Type = 'file';
    sourceInfo.Location = char(fileName);
else
    error('PhyGESS:UnsupportedDataSource', ...
        'dataSource must be a table, timetable, numeric matrix, CSV file, or Excel file.');
end

if firstColumnIsTime && width(tableData) > 1
    tableData = tableData(:, 2:end);
end

isNumericColumn = varfun(@isnumeric, tableData, 'OutputFormat', 'uniform');
if ~all(isNumericColumn)
    badNames = tableData.Properties.VariableNames(~isNumericColumn);
    error('PhyGESS:NonNumericTelemetry', ...
        'Telemetry columns must be numeric after timestamp removal. Non-numeric columns: %s', ...
        strjoin(badNames, ', '));
end

telemetry = double(table2array(tableData));
end
