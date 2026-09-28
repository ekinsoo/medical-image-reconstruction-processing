# Medical Image Reconstruction and Processing

Coursework for **EEE 475: Medical Image Reconstruction and Processing** at Özyeğin University. This repository collects homework submissions and a course project evaluating a pre-trained model for accelerated MRI reconstruction.

**Author:** Ozgur Ekin Sonmez

## Repository layout

- `homeworks/HW1` — Fourier-domain MRI reconstruction and sampling experiments.
- `homeworks/HW2` — radial and spiral trajectories, gridding, and density compensation.
- `homeworks/HW3` — system-matrix reconstruction and iterative methods.
- `homeworks/HW4` — parallel MRI experiments, including SENSE/GRAPPA methods.
- `homeworks/HW5` — compressed-sensing MRI and wavelet-based reconstruction.
- `project` — ReconFormer evaluation notebook, experiment notes, results, and presentation.

Each homework folder contains the submitted report and, where available, the code or notebook used for the work. Some folders also contain assignment PDFs and supplementary files.

## Course project: ReconFormer evaluation

The project runs inference with published pre-trained ReconFormer weights on 30 scans from the fastMRI single-coil knee validation set. It compares acceleration factors 4 and 8 using PSNR, SSIM, and NMSE, and includes example visual comparisons.

This is an evaluation of an existing model; the model was not trained as part of this project. The dataset, pretrained weights, and upstream ReconFormer source code are not included. The notebook and workflow notes document the evaluation and compatibility changes used in the Colab run.

Project files:

- [Evaluation notebook](project/ReconFormer_Eval.ipynb)
- [Workflow and implementation notes](project/ReconFormer_WorkflowUpdated.md)
- [Qualitative results](project/qualitativeResults.pdf)
- [Quantitative results](project/quantitativeResults.pdf)
- [Presentation](project/ReconFormer_Professor_Ready_Presentation.pptx)

## Tools

MATLAB, Python, Jupyter, NumPy, SciPy, PyTorch, scikit-image, and LaTeX.
