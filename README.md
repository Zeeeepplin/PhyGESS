# PhyGESS: Physics-Informed Generative Estimation for Substation Systems

**PhyGESS** is an open-source MATLAB toolbox designed for robust, physically-bounded thermal state estimation of power grid distribution transformers. The framework systematically bridges generative machine learning and scientific computing to solve two critical bottlenecks in grid asset analytics: telemetry data scarcity and PINN training instability.

### Core Architecture
1. **Generative State-Space Densification:** Leverages an upstream Gaussian-Bernoulli Restricted Boltzmann Machine (GB-RBM) to expand sparse, low-resolution substation logs into dense, continuous multi-modal telemetry matrices—preserving data integrity via a strict decoupled training pipeline.
2. **Pathology-Free SciML Engine:** Embeds a Physics-Informed Neural Network (PINN) regulated by a **Dynamic Loss Weight Annealing** controller. Built natively on MATLAB's automatic differentiation (`dlgradient`) graph, it eliminates *Saturated Optimization Divergence* by continuously balancing empirical data gradients against first-order differential thermodynamic lag laws ($\tau$).
