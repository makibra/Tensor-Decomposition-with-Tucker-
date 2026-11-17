# Tensor Decomposition with Tucker — COVID-19 Systems Serology

A practical project exploring **Tucker tensor decomposition** on COVID-19 systems-serology data. The objective is to compress a three-way tensor and investigate latent patterns linking patients, SARS-CoV-2 antigens, and antibody receptors.

> **Project status:** The notebook currently implements data loading and exploration, NumPy implementations of the mode-\(n\) product and Tucker reconstruction, and a grid search over multilinear ranks. Complete the rank-selection, random-initialization stability study, and factor/core interpretation sections with the results produced by your own execution before publishing final numerical conclusions.

## Dataset

The dataset is loaded with `load_covid19_serology()` from TensorLy. It is represented as a third-order tensor:

\[
X \in \mathbb{R}^{438 \times 6 \times 11}
\]

| Mode | Size | Description |
|---|---:|---|
| Samples | 438 | Patients, labelled by disease outcome |
| Antigens | 6 | S, RBD, N, S1, S2, S1 Trimer |
| Receptors | 11 | Antibody isotypes and Fc receptors |

The tensor entry \(X_{i,j,k}\) represents the measured binding intensity for patient \(i\), antigen \(j\), and receptor \(k\).

The dataset is associated with the COVID-19 systems-serology study by Zohar et al. (2020), and its tensor formulation is described by Tan et al. (2021). See **References** below.

## Method: Tucker decomposition

Tucker decomposition approximates a tensor by a smaller core tensor and one factor matrix per mode:

\[
X \approx \mathcal{G} \times_1 A \times_2 B \times_3 C
\]

where:

- \(\mathcal{G}\) is the core tensor;
- \(A\) contains the sample-mode factors;
- \(B\) contains the antigen-mode factors;
- \(C\) contains the receptor-mode factors;
- \((r_1,r_2,r_3)\) are the multilinear ranks.

The number of stored parameters is:

\[
P = r_1r_2r_3 + 438r_1 + 6r_2 + 11r_3
\]

The relative reconstruction error is:

\[
E = \frac{\lVert X-\widehat{X}\rVert_F}{\lVert X\rVert_F}
\]

## Project contents

The notebook is designed to cover the following steps:

1. **Load and inspect the data**
   - Load the TensorLy dataset.
   - Display tensor dimensions, antigen/receptor labels, and patient outcome counts.
   - Check missing/infinite values and inspect the mean and standard deviation.

2. **Implement tensor operations with NumPy**
   - Implement the mode-\(n\) product using unfolding, matrix multiplication, and folding.
   - Implement reconstruction for a three-way Tucker decomposition.
   - Compare the manual reconstruction with `tensorly.tucker_to_tensor`.

3. **Explore multilinear ranks**
   - Evaluate the requested grid:
     - \(r_1 \in \{2,4,6,8\}\)
     - \(r_2 \in \{2,3,4\}\)
     - \(r_3 \in \{2,4,6\}\)
   - Store each decomposition, its relative reconstruction error, and its parameter count.

4. **Investigate warnings and repeated errors**
   - Compare the requested rank combinations.
   - Interpret TensorLy warnings in relation to the effective numerical rank and the algorithm's convergence.

5. **Select a rank and study stability**
   - Plot reconstruction error against parameter count.
   - Select a rank using the accuracy/complexity trade-off.
   - Repeat Tucker decomposition 20 times with random initializations and compare the errors with the SVD initialization.

6. **Interpret the factors and core**
   - Plot heatmaps for antigen and receptor factor matrices.
   - Visualize slices of the core tensor.
   - Discuss latent patterns cautiously; latent components are not automatically biological categories.

## Installation

Python 3.10 or newer is recommended. Install the required packages:

```bash
python -m pip install tensorly numpy pandas matplotlib jupyter
```

## Run the notebook

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Start Jupyter:

```bash
jupyter notebook
```

Open the project notebook and run the cells in order. The dataset loader may need an internet connection the first time it retrieves the dataset.

## Suggested repository structure

```text
.
├── README.md
└── Tucker_COVID.ipynb

```


## Notes on interpretation

A lower reconstruction error indicates a closer numerical approximation, but it does not by itself establish biological significance. Factor signs and component ordering can vary between decompositions. Interpret factor loadings together with the core tensor and the dataset context, and avoid treating latent components as direct clinical or causal explanations.

## References

- Zohar et al. (2020), *Cell* — study of the antibody response in COVID-19 patients.
- Tan et al. (2021), *Molecular Systems Biology* — tensor formulation of the systems-serology dataset.
- [TensorLy documentation](https://tensorly.org/stable/)

## Author

**Ibrahim Makhlouf**  
Master 2 — Artificial Intelligence and Machine Learning  
Aix-Marseille Université / École Centrale Méditerranée
