% PhyGESS Physics-Informed Generative Estimation for Substation Systems.
%
%   result = PhyGESS.trainGBRBMPINN(dataSource) trains a GB-RBM from
%   substation telemetry, synthesizes additional telemetry, and trains a
%   PINN on the generated current-to-temperature relationship.
%
%   [rbm, info] = PhyGESS.trainGBRBM(X) trains a Gaussian-Bernoulli
%   RBM from normalized data arranged as [numVisible x numSamples].
%
%   [data, normalizedData] = PhyGESS.generateGBRBMSamples(rbm, N, mu, sigma)
%   draws N physical telemetry samples from a trained GB-RBM.
%
%   [model, info] = PhyGESS.trainPINN(inputs, targets) trains the PINN
%   component using data loss plus a Joule-heating physics residual.
%
%   predictions = PhyGESS.predictPINN(model, inputs) evaluates a
%   trained PINN in physical units.
%
%   metrics = PhyGESS.validatePINN(model, inputs, targets) evaluates
%   RMSE, MAE, bias, and residuals.
