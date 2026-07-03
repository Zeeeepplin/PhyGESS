function predictions = predictPINN(model, inputs)
%PREDICTPINN Predict physical target values from a trained PINN model.

validateattributes(model, {'struct'}, {'scalar'}, mfilename, 'model', 1);
validateattributes(inputs, {'numeric'}, {'vector', 'real'}, mfilename, 'inputs', 2);

if isfield(model, 'Network')
    net = model.Network;
elseif isfield(model, 'dlnet')
    net = model.dlnet;
else
    error('PhyGESS:InvalidPINNModel', 'model must contain a Network field.');
end

requiredFields = {'InputMean', 'InputStd', 'TargetMean', 'TargetStd'};
for idx = 1:numel(requiredFields)
    if ~isfield(model, requiredFields{idx})
        error('PhyGESS:InvalidPINNModel', 'model is missing the %s field.', requiredFields{idx});
    end
end

inputShape = size(inputs);
inputs = inputs(:);
normalizedInputs = (inputs - model.InputMean) / model.InputStd;
networkInput = dlarray(normalizedInputs', 'CB');

normalizedPredictions = predict(net, networkInput);
predictions = (double(extractdata(normalizedPredictions))' .* model.TargetStd) + model.TargetMean;
predictions = reshape(predictions, inputShape);
end
