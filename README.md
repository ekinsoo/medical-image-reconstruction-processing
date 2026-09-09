# Medical Image Reconstruction and Processing

Coursework and a course project for EEE 475 Medical Image Reconstruction and Processing. The repository includes homework implementations and reports covering MRI sampling, k-space processing, non-Cartesian reconstruction, system-matrix reconstruction, parallel imaging, compressed sensing, and a deep-learning evaluation project for accelerated MRI.

Author: Ozgur Ekin Sonmez

## Contents

| Folder | Topic | Summary |
| --- | --- | --- |
| `homeworks/HW1` | Fourier and k-space reconstruction | Basic MR image reconstruction, sampling behavior, zero filling, and POCS-style experiments. |
| `homeworks/HW2` | Non-Cartesian reconstruction | MATLAB implementations for radial/spiral trajectories, gridding, density compensation, and filtered backprojection. |
| `homeworks/HW3` | Linear-system reconstruction | System-matrix based reconstruction, singular-value analysis, TSVD, and iterative methods. |
| `homeworks/HW4` | Parallel MRI | Multi-coil reconstruction with SENSE/GRAPPA-style experiments and g-factor analysis. |
| `homeworks/HW5` | Compressed sensing MRI | L1-SPIRiT and wavelet-based reconstruction experiments in MATLAB. |
| `project` | ReconFormer evaluation | Evaluation workflow for a pre-trained ReconFormer model on accelerated MRI reconstruction. |

## Project Highlight: ReconFormer Evaluation

The course project evaluates ReconFormer, a transformer-based model for accelerated MRI reconstruction, using pre-trained weights and a subset of the fastMRI single-coil knee validation data. The work focuses on inference, compatibility updates for modern PyTorch, metric reporting, and qualitative comparison figures for different acceleration factors.

Key files:

- `project/ReconFormer_Eval.ipynb`: evaluation notebook.
- `project/ReconFormer_WorkflowUpdated.md`: reproducible workflow and implementation notes.
- `project/qualitativeResults.pdf`: qualitative reconstruction comparisons.
- `project/quantitativeResults.pdf`: PSNR, SSIM, and NMSE summary.
- `project/ReconFormer_Professor_Ready_Presentation.pptx`: presentation deck.

## Tech Stack

MATLAB, Python, Jupyter, NumPy, PyTorch, fastMRI-style MRI data processing, LaTeX.

## Notes

Large raw datasets, `.mat` measurement files, fastMRI `.h5` data, generated output folders, archives, compiled MATLAB binaries, and old Git metadata are intentionally excluded. The repository keeps the code, notebooks, reports, and project documentation needed to understand the work without making the repo unnecessarily large.
