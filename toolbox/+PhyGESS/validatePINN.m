function metrics = validatePINN(model, inputs, targets)
%VALIDATEPINN Evaluate a trained PINN against physical target data.

validateattributes(inputs, {'numeric'}, {'vector', 'real'}, mfilename, 'inputs', 2);
validateattributes(targets, {'numeric'}, {'vector', 'real'}, mfilename, 'targets', 3);

inputs = inputs(:);
targets = targets(:);
if numel(inputs) ~= numel(targets)
    error('PhyGESS:ValidationSizeMismatch', 'inputs and targets must have the same number of elements.');
end

validRows = isfinite(inputs) & isfinite(targets);
if ~all(validRows)
    warning('PhyGESS:ValidationRemovedRows', 'Removing %d rows with NaN or Inf values.', nnz(~validRows));
    inputs = inputs(validRows);
    targets = targets(validRows);
end

predictions = PhyGESS.predictPINN(model, inputs);
residuals = targets - predictions(:);

metrics = struct();
metrics.RMSE = sqrt(mean(residuals.^2));
metrics.MAE = mean(abs(residuals));
metrics.Bias = mean(residuals);
metrics.NumSamples = numel(targets);
metrics.Predictions = predictions(:);
metrics.Targets = targets;
metrics.Residuals = residuals;
end
