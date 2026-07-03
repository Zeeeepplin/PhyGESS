# PhyGESS Toolbox

PhyGESS is short for Physics-Informed Generative Estimation for Substation
Systems. This toolbox combines the Gaussian-Bernoulli RBM workflow from
`GBRBM.m` with the physics-informed neural network workflow from `NewPINN.m`.

The toolbox .mltbx file is located in the /dist folder


The main entry point is:

```matlab
result = PhyGESS.trainGBRBMPINN("datalogsheet.xlsx");
```

The combined pipeline:

1. Reads substation telemetry from a numeric matrix, table, CSV, or Excel file.
2. Normalizes the physical telemetry channels.
3. Trains a Gaussian-Bernoulli RBM.
4. Synthesizes additional physical telemetry.
5. Trains a PINN on the generated current-to-temperature relationship.
6. Validates the PINN against the original telemetry.

## Quick Start

From MATLAB:

```matlab
addpath("C:\DigitalTwin\PhyGESSToolbox\toolbox")

result = PhyGESS.trainGBRBMPINN("C:\DigitalTwin\datalogsheet.xlsx", ...
    "InputColumn", 2, ...
    "TargetColumn", 4, ...
    "NumSyntheticSamples", 10000, ...
    "GBRBMEpochs", 500, ...
    "PINNEpochs", 600, ...
    "SavePath", "C:\DigitalTwin\Combined_GBRBM_PINN_Result.mat");

disp(result.Validation)
```

`InputColumn` and `TargetColumn` are counted after the first timestamp column
is removed. The defaults match the existing scripts: column 2 is current and
column 4 is transformer temperature.

## Separate Components

Train only the GB-RBM:

```matlab
T = readtable("C:\DigitalTwin\datalogsheet.xlsx");
telemetry = table2array(T(:, 2:end));
mu = mean(telemetry, 1);
sigma = std(telemetry, 0, 1);
sigma(sigma == 0) = 1;

[rbm, rbmInfo] = PhyGESS.trainGBRBM(((telemetry - mu) ./ sigma)');
syntheticTelemetry = PhyGESS.generateGBRBMSamples(rbm, 10000, mu, sigma);
```

Train only the PINN:

```matlab
[model, pinnInfo] = PhyGESS.trainPINN( ...
    syntheticTelemetry(:, 2), syntheticTelemetry(:, 4));

metrics = PhyGESS.validatePINN(model, telemetry(:, 2), telemetry(:, 4));
```

## Requirements

- MATLAB R2024b or newer
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox for workflows that use kernel
  density estimation outside this package
