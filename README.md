# Medical Image Reconstruction and Processing

Coursework for **EEE 475/575: Medical Image Reconstruction and Processing** at Bilkent University. The repository contains homework submissions and the team report for a course project on accelerated MRI reconstruction.

## Repository layout

- `homeworks/HW1` — Fourier-domain reconstruction and k-space sampling.
- `homeworks/HW2` — radial and spiral sampling, gridding, and density compensation.
- `homeworks/HW3` — system-matrix reconstruction and iterative methods.
- `homeworks/HW4` — parallel MRI, including SENSE and GRAPPA experiments.
- `homeworks/HW5` — compressed-sensing MRI and wavelet-based reconstruction.
- `project` — the course project report.

## Course project

**ReconFormer: Evaluation of a Recurrent Transformer for Accelerated MRI Reconstruction** evaluates the authors' pre-trained model on 30 scans (1,059 slices) from the fastMRI single-coil knee validation set. The report compares acceleration factors 4 and 8 using PSNR, SSIM, and NMSE. It also describes the evaluation setup, compatibility changes, qualitative results, limitations, individual contributions, and generative AI disclosure.

The project did not train the model or independently run the D5C5 baseline; those published baseline values are cited in the report. The dataset, checkpoints, and upstream ReconFormer code are not included in this repository.

- [MIR project report](project/MIR_Project.pdf)

## Tools used

MATLAB, Python, Jupyter, PyTorch, and LaTeX.
