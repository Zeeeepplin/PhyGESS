function toolboxFile = buildToolbox()
%BUILDTOOLBOX Package the PhyGESS Toolbox as an .mltbx file.

root = fileparts(mfilename('fullpath'));
distFolder = fullfile(root, 'dist');
if ~isfolder(distFolder)
    mkdir(distFolder);
end

toolboxFile = fullfile(distFolder, 'PhyGESSToolbox.mltbx');
identifier = 'PhyGESSToolbox';

opts = matlab.addons.toolbox.ToolboxOptions(root, identifier);
opts.ToolboxName = 'PhyGESS Toolbox';
opts.ToolboxVersion = '1.0.0';
opts.Summary = 'Physics-Informed Generative Estimation for Substation Systems.';
opts.Description = [ ...
    'PhyGESS combines Gaussian-Bernoulli RBM telemetry synthesis with ', ...
    'physics-informed neural network state estimation for substation ', ...
    'digital twin workflows.'];
opts.AuthorName = 'PhyGESS Developers';
opts.MinimumMatlabRelease = 'R2024b';
opts.OutputFile = toolboxFile;

opts.ToolboxFiles = [
    string(fullfile(root, 'README.md'))
    string(fullfile(root, 'buildToolbox.m'))
    localListFiles(fullfile(root, 'toolbox'))
    localListFiles(fullfile(root, 'examples'))];

opts.ToolboxMatlabPath = [
    string(fullfile(root, 'toolbox'))
    string(fullfile(root, 'examples'))];

matlab.addons.toolbox.packageToolbox(opts);
fprintf('Packaged toolbox: %s\n', toolboxFile);
end

function files = localListFiles(folder)
items = dir(fullfile(folder, '**', '*'));
items = items(~[items.isdir]);
files = string(arrayfun(@(item) fullfile(item.folder, item.name), items, 'UniformOutput', false));
files = files(:);
end
