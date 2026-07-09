function [model, info] = trainPINN(inputs, targets, varargin)
%TRAINPINN Train a physics-informed neural network.
%
%   model = PhyGESS.trainPINN(inputs, targets) trains a dlnetwork that
%   maps one physical input channel, such as current, to one physical target
%   channel, such as transformer temperature.
%
%   The loss combines supervised data error with a Joule-heating residual:
%       tau*dT/dX + T - (k*I^2 + ambient)

validateattributes(inputs, {'numeric'}, {'vector', 'real'}, mfilename, 'inputs', 1);
validateattributes(targets, {'numeric'}, {'vector', 'real'}, mfilename, 'targets', 2);

inputs = inputs(:);
targets = targets(:);
if numel(inputs) ~= numel(targets)
    error('PhyGESS:PINNSizeMismatch', 'inputs and targets must have the same number of elements.');
end

validRows = isfinite(inputs) & isfinite(targets);
if ~all(validRows)
    warning('PhyGESS:PINNRemovedRows', 'Removing %d rows with NaN or Inf values.', nnz(~validRows));
    inputs = inputs(validRows);
    targets = targets(validRows);
end

p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'MaxEpochs', 600, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'LearningRate', 0.005, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'LayerWidth', 20, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'NumHiddenLayers', 2, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'PhysicsWeight', 0.1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'PhysicsK', 0.0002, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'AmbientTemperature', 30, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'ThermalTau', 15, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'UseDerivativePhysics', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Verbose', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'ShowProgress', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'RandomSeed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});

opts = p.Results;
opts.MaxEpochs = round(opts.MaxEpochs);
opts.LayerWidth = round(opts.LayerWidth);
opts.NumHiddenLayers = round(opts.NumHiddenLayers);
opts.Verbose = logical(opts.Verbose);
opts.ShowProgress = logical(opts.ShowProgress);
opts.UseDerivativePhysics = logical(opts.UseDerivativePhysics);

if ~isempty(opts.RandomSeed)
    oldState = rng(opts.RandomSeed);
    cleanup = onCleanup(@() rng(oldState));
end

muInput = mean(inputs);
sigmaInput = std(inputs);
if sigmaInput == 0
    sigmaInput = 1;
end

muTarget = mean(targets);
sigmaTarget = std(targets);
if sigmaTarget == 0
    sigmaTarget = 1;
end

normalizedInputs = (inputs - muInput) / sigmaInput;
normalizedTargets = (targets - muTarget) / sigmaTarget;

X = dlarray(normalizedInputs', 'CB');
Y = dlarray(normalizedTargets', 'CB');

layers = localBuildLayers(opts.LayerWidth, opts.NumHiddenLayers);
net = dlnetwork(layerGraph(layers));

trailingAvg = [];
trailingAvgSq = [];

totalLossHistory = zeros(opts.MaxEpochs, 1);
dataLossHistory = zeros(opts.MaxEpochs, 1);
physicsLossHistory = zeros(opts.MaxEpochs, 1);

monitor = [];
if opts.ShowProgress
    monitor = trainingProgressMonitor( ...
        'Metrics', ["TotalLoss", "DataLoss", "PhysicsLoss"], ...
        'Info', 'Epoch', ...
        'XLabel', 'Iteration');
    monitor.Status = 'Training PINN';
end

scales = struct( ...
    'InputMean', muInput, ...
    'InputStd', sigmaInput, ...
    'TargetMean', muTarget, ...
    'TargetStd', sigmaTarget);

physics = struct( ...
    'Weight', opts.PhysicsWeight, ...
    'K', opts.PhysicsK, ...
    'AmbientTemperature', opts.AmbientTemperature, ...
    'ThermalTau', opts.ThermalTau, ...
    'UseDerivativePhysics', opts.UseDerivativePhysics);

for epoch = 1:opts.MaxEpochs
    [loss, dataLoss, physicsLoss, gradients] = dlfeval(@localModelLoss, net, X, Y, scales, physics);

    [net, trailingAvg, trailingAvgSq] = adamupdate( ...
        net, gradients, trailingAvg, trailingAvgSq, epoch, opts.LearningRate);

    totalLossHistory(epoch) = double(extractdata(loss));
    dataLossHistory(epoch) = double(extractdata(dataLoss));
    physicsLossHistory(epoch) = double(extractdata(physicsLoss));

    if opts.ShowProgress
        recordMetrics(monitor, epoch, ...
            'TotalLoss', totalLossHistory(epoch), ...
            'DataLoss', dataLossHistory(epoch), ...
            'PhysicsLoss', physicsLossHistory(epoch));
        updateInfo(monitor, Epoch=epoch);
        monitor.Progress = (epoch / opts.MaxEpochs) * 100;
    end

    if opts.Verbose && (epoch == 1 || mod(epoch, 100) == 0 || epoch == opts.MaxEpochs)
        fprintf('PINN epoch %d/%d | total %.4f | data %.4f | physics %.4f\n', ...
            epoch, opts.MaxEpochs, totalLossHistory(epoch), dataLossHistory(epoch), physicsLossHistory(epoch));
    end
end

model = struct();
model.Network = net;
model.InputMean = muInput;
model.InputStd = sigmaInput;
model.TargetMean = muTarget;
model.TargetStd = sigmaTarget;
model.Physics = physics;
model.TrainingOptions = opts;

info = table((1:opts.MaxEpochs)', totalLossHistory, dataLossHistory, physicsLossHistory, ...
    'VariableNames', {'Epoch', 'TotalLoss', 'DataLoss', 'PhysicsLoss'});
end

function layers = localBuildLayers(layerWidth, numHiddenLayers)
layers = featureInputLayer(1, 'Name', 'input', 'Normalization', 'none');
for idx = 1:numHiddenLayers
    layers = [
        layers
        fullyConnectedLayer(layerWidth, 'Name', sprintf('fc%d', idx))
        tanhLayer('Name', sprintf('tanh%d', idx))]; %#ok<AGROW>
end
layers = [
    layers
    fullyConnectedLayer(1, 'Name', 'output')];
end

function [totalLoss, lossData, lossPhysics, gradients] = localModelLoss(net, X, target, scales, physics)
prediction = forward(net, X);

lossData = mean((prediction - target).^2, 'all');

inputPhysical = (X * scales.InputStd) + scales.InputMean;
targetPhysical = (prediction * scales.TargetStd) + scales.TargetMean;
physicsTarget = (physics.K * (inputPhysical.^2)) + physics.AmbientTemperature;

if physics.UseDerivativePhysics
    dTdn = dlgradient(sum(targetPhysical, 'all'), X, 'EnableHigherDerivatives', true);
    dTdX = dTdn ./ scales.InputStd;
    residual = (physics.ThermalTau * dTdX) + targetPhysical - physicsTarget;
else
    residual = targetPhysical - physicsTarget;
end

lossPhysics = mean((residual ./ scales.TargetStd).^2, 'all');
totalLoss = lossData + (physics.Weight * lossPhysics);

gradients = dlgradient(totalLoss, net.Learnables);
end
