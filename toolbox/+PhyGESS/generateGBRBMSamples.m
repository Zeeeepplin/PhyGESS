function [syntheticData, normalizedData] = generateGBRBMSamples(rbm, numSamples, muData, sigmaData, varargin)
%GENERATEGBRBMSAMPLES Generate physical telemetry samples from a GB-RBM.
%
%   data = PhyGESS.generateGBRBMSamples(rbm, N, mu, sigma) generates N
%   samples and maps them from normalized units back to physical units using
%   the supplied training mean and standard deviation vectors.

validateattributes(rbm, {'struct'}, {'scalar'}, mfilename, 'rbm', 1);
validateattributes(numSamples, {'numeric'}, {'scalar', 'integer', 'positive'}, mfilename, 'numSamples', 2);

if ~isfield(rbm, 'Weights') || ~isfield(rbm, 'BiasVisible') || ~isfield(rbm, 'BiasHidden')
    error('PhyGESS:InvalidGBRBM', 'rbm must contain Weights, BiasVisible, and BiasHidden fields.');
end

numVisible = size(rbm.Weights, 1);
if nargin < 3 || isempty(muData)
    muData = zeros(1, numVisible);
end
if nargin < 4 || isempty(sigmaData)
    sigmaData = ones(1, numVisible);
end

validateattributes(muData, {'numeric'}, {'row', 'numel', numVisible, 'real', 'finite'}, mfilename, 'muData', 3);
validateattributes(sigmaData, {'numeric'}, {'row', 'numel', numVisible, 'real', 'finite'}, mfilename, 'sigmaData', 4);

p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'BurnInSteps', 30, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'RandomSeed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});

if ~isempty(p.Results.RandomSeed)
    oldState = rng(p.Results.RandomSeed);
    cleanup = onCleanup(@() rng(oldState));
end

weights = rbm.Weights;
bVisible = rbm.BiasVisible;
bHidden = rbm.BiasHidden;
numHidden = size(weights, 2);

hiddenStates = double(rand(numHidden, numSamples) > 0.5);

for step = 1:round(p.Results.BurnInSteps)
    visibleContinuous = weights * hiddenStates + bVisible;
    hiddenProb = localSigmoid(weights' * visibleContinuous + bHidden);
    hiddenStates = double(rand(size(hiddenProb)) < hiddenProb);
end

normalizedData = (weights * hiddenStates + bVisible)';
syntheticData = (normalizedData .* sigmaData) + muData;
end

function y = localSigmoid(x)
x = max(min(x, 50), -50);
y = 1 ./ (1 + exp(-x));
end
