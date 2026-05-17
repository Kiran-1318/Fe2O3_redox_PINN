# Fe₂O₃ Redox Kinetics — Physics-Informed Neural Network and Operator Learning

Physics-Informed Neural Network (PINN) and Physics-Informed DeepONet (PI-DeepONet)
for identifying Fe₂O₃ reduction kinetics from sparse isothermal TGA data in
chemical looping hydrogen generation (CLHG).

**Preprint:** *submitted to ChemRxiv — link will be added upon publication*

---

## Overview

Fe₂O₃ oxygen carriers are cyclically reduced by H₂ in the fuel reactor of CLHG
systems and re-oxidised by steam to produce pure H₂. The reduction kinetics —
governed by the Arrhenius shrinking core model — are critical inputs for reactor
design. This repository implements a two-stage physics-informed machine learning
pipeline applied to experimental TGA data from Wang et al. (2023).

**Data source:** Wang H. et al. (2023), "Multistep kinetic study of Fe₂O₃
reduction by H₂ based on isothermal thermogravimetric analysis data
deconvolution," *Int. J. Hydrogen Energy*, 48, 16601–16613.
Data digitised from Figures 4–6 at 750, 800, 850, 900, 950°C using
WebPlotDigitizer. Baseline correction applied: X = (X_raw − X₀)/(1 − X₀).

---

## Notebooks

### Notebook 1 — Inverse PINN: Kinetic Parameter Identification

`1_Fe2O3_Inverse_PINN.ipynb`

**Governing equation:**
dX/dt = A · exp(−Eₐ/RT) · (1−X)^(2/3),   X(0) = 0

**What it does:** Simultaneously identifies the pre-exponential factor A and
apparent activation energy Eₐ as trainable variables from 24 sparse isothermal
TGA observations (8 points × 3 training temperatures).

**Loss function:**

L = λ_phys · L_physics + λ_data · L_data + λ_ic · L_ic

**Key results:**

| Temperature | R² | Status |
|-------------|-----|--------|
| 750°C | 0.9988 | Training |
| 850°C | 0.9981 | Training |
| 900°C | 0.9974 | Training |
| 800°C | 0.9675 | Validation (interpolation) |
| 950°C | 0.2712 | Validation (extrapolation) — expected failure |

**Recovered parameters:**
- Apparent Eₐ = **24.07 kJ/mol** (consistent with Wang 2023: 10.3, 26.7, 24.8 kJ/mol for three steps)
- A = 5.87×10⁻² s⁻¹

**Framework:** Pytorch

---

### Notebook 2 — PI-DeepONet: Physics-Informed Operator Learning

`2_PI_DeepONet_Fe2O3.ipynb`

**What it does:** Learns the operator G: T → X(t) — mapping temperature to
the full conversion trajectory — using a DeepONet with physics residual loss
enforced at 10,000 collocation points across the full temperature domain.

**Architecture:**
- Branch net: T_norm → ℝ¹⁰⁰  [1→64→64→100], tanh
- Trunk net:  t_norm → ℝ¹⁰⁰  [1→64→64→100→], tanh
- Output: X(t|T) = branch(T) · trunk(t) + bias

**Training data:** 24 sparse corrected TGA points (8 per training temperature).
A and Eₐ fixed from Notebook 1. Physics loss uses recovered parameters.

**Key results:**

| Method | 750°C | 850°C | 900°C | 800°C | 950°C | Physics | Retrain per T |
|--------|-------|-------|-------|-------|-------|---------|---------------|
| Inverse PINN (NB1) | 0.999 | 0.998 | 0.997 | 0.968 | 0.271 | Yes | Yes |
| Data-only DeepONet | 0.999 | 0.997 | 0.996 | 0.995 | 0.524 | No | No |
| **PI-DeepONet** | **0.996** | **0.991** | **0.982** | **0.982** | **0.978** | **Yes** | **No** |

**R² values. Training: 750, 850, 900°C. Validation: 800°C (interpolation), 950°C (extrapolation).**

**Key finding:** Physics-constrained operator learning achieves R²=0.978 at
950°C — a temperature entirely absent from training — versus R²=0.524 for a
data-only operator and R²=0.271 for a single-condition inverse PINN.

**Framework:** PyTorch (raw implementation)

---

## Pipeline
Wang 2023 TGA data (5 temperatures, 750–950°C)
│
▼
Notebook 1 — Inverse PINN
Input:  24 sparse TGA points (8 per temp × 3 training temps)
Output: A = 5.87×10⁻² s⁻¹, Eₐ = 24.07 kJ/mol
│
▼ (A, Eₐ fixed as physics constants)
Notebook 2 — PI-DeepONet
Input:  24 sparse points + physics residual at 10,000 collocation points
Output: Operator G: T → X(t), valid for T ∈ [750, 950°C]
950°C R² = 0.978 (unseen temperature, no retraining)

---

## Physical context

In chemical looping hydrogen generation, Fe₂O₃ is reduced in a fuel reactor
by H₂ produced from biomass or fossil fuel pyrolysis. The three-step reduction:
Fe₂O₃ → Fe₃O₄ → FeO → Fe
is approximated by a lumped shrinking core model for reactor-scale modelling.
The shrinking core ODE describes how the unreacted particle core shrinks as
the product layer grows, with conversion rate proportional to the remaining
surface area (∝ (1−X)^(2/3) for spherical particles).

The apparent activation energy (Eₐ = 24.07 kJ/mol) represents the
weighted average of the three step-wise values reported by Wang et al. (2023)
and is the relevant kinetic parameter for reactor residence time and solid
inventory calculations.

---

## Requirements

torch>=2.0.0
numpy
pandas
matplotlib
scikit-learn
scipy

No DeepXDE or TensorFlow required. Both notebooks run on CPU or GPU (Colab T4).

---

## Related repositories

[1D-Heat-Equation-PINN](https://github.com/Kiran-1318/1D-Heat-Equation-PINN)
— Physics-Informed Neural Network for the 1D heat equation. Rel L2: 0.99%.

[DeepXDE-Chemical-Looping-Problems](https://github.com/Kiran-1318/DeepXDE-Chemical-Looping-Problems)
— 10 original PINN problems: forward ODEs → inverse Arrhenius identification
→ multi-cycle redox → physics-data hybrid (preprint template).

[NeuralOperator-Chemical-Looping-Problems](https://github.com/Kiran-1318/NeuralOperator-Chemical-Looping-Problems)
— 5 neural operator problems: DeepONet, FNO from scratch, PI-DeepONet.
PI-DeepONet achieves 30× better generalisation than data-only at unseen temperatures.

---

## Citation

If you use this code or data, please cite:
Kiran, T. (2026). Physics-Informed Neural Networks and Operator Learning
for Fe₂O₃ Reduction Kinetics Identification from Sparse Thermogravimetric Data.
ChemRxiv. [DOI will be added upon publication]

And the original TGA data source:

Wang, H. et al. (2023). Multistep kinetic study of Fe₂O₃ reduction by H₂
based on isothermal thermogravimetric analysis data deconvolution.
International Journal of Hydrogen Energy, 48, 16601–16613.

---

## Author

**Kiran Thammina**
M.Tech Energy Systems Engineering, IIT Bombay (CPI 9.84, Best Thesis Award)
GitHub: [github.com/Kiran-1318](https://github.com/Kiran-1318)
