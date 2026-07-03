function [rbm, info] = trainGBRBM(X, varargin)
%TRAINGBRBM Train a Gaussian-Bernoulli RBM for continuous telemetry.
%
%   rbm = PhyGESS.trainGBRBM(X) trains a Gaussian-Bernoulli RBM using
%   one-step contrastive divergence. X must be normalized and arranged as
%   [numVisible x numSamples].
%
%   [rbm, info] = PhyGESS.trainGBRBM(..., Name=Value) returns the
%   trained model and per-epoch reconstruction RMSE.

validateattributes(X, {'numeric'}, {'2d', 'nonempty', 'real', 'finite'}, mfilename, 'X', 1);

p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'NumHiddenUnits', 50, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'MaxEpochs', 500, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'LearningRate', 0.005, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'WeightPenalty', 0.001, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'Momentum', 0.9, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'Verbose', true, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});

numHidden = round(p.Results.NumHiddenUnits);
epochs = round(p.Results.MaxEpochs);
eta = p.Results.LearningRate;
lambda = p.Results.WeightPenalty;
alpha = p.Results.Momentum;
verbose = logical(p.Results.Verbose);

[numVisible, numSamples] = size(X);

weights = 0.01 * randn(numVisible, numHidden);
bVisible = zeros(numVisible, 1);
bHidden = zeros(numHidden, 1);

velocityWeights = zeros(size(weights));
velocityVisible = zeros(size(bVisible));
velocityHidden = zeros(size(bHidden));

rmseHistory = zeros(epochs, 1);

for epoch = 1:epochs
    hiddenProbPositive = localSigmoid(weights' * X + bHidden);
    hiddenStates = double(rand(size(hiddenProbPositive)) < hiddenProbPositive);
    positiveAssociations = X * hiddenProbPositive';

    visibleNegative = weights * hiddenStates + bVisible;
    hiddenProbNegative = localSigmoid(weights' * visibleNegative + bHidden);
    negativeAssociations = visibleNegative * hiddenProbNegative';

    deltaWeights = ((positiveAssociations - negativeAssociations) / numSamples) - (lambda * weights);
    deltaVisible = mean(X - visibleNegative, 2);
    deltaHidden = mean(hiddenProbPositive - hiddenProbNegative, 2);

    velocityWeights = (alpha * velocityWeights) + (eta * deltaWeights);
    velocityVisible = (alpha * velocityVisible) + (eta * deltaVisible);
    velocityHidden = (alpha * velocityHidden) + (eta * deltaHidden);

    weights = weights + velocityWeights;
    bVisible = bVisible + velocityVisible;
    bHidden = bHidden + velocityHidden;

    rmseHistory(epoch) = sqrt(mean((X(:) - visibleNegative(:)).^2));

    if verbose && (epoch == 1 || mod(epoch, 50) == 0 || epoch == epochs)
        fprintf('GB-RBM epoch %d/%d | RMSE %.4f\n', epoch, epochs, rmseHistory(epoch));
    end
end

rbm = struct();
rbm.Weights = weights;
rbm.BiasVisible = bVisible;
rbm.BiasHidden = bHidden;
rbm.NumVisibleUnits = numVisible;
rbm.NumHiddenUnits = numHidden;
rbm.TrainingOptions = p.Results;

info = table((1:epochs)', rmseHistory, 'VariableNames', {'Epoch', 'ReconstructionRMSE'});
end

function y = localSigmoid(x)
x = max(min(x, 50), -50);
y = 1 ./ (1 + exp(-x));
end
