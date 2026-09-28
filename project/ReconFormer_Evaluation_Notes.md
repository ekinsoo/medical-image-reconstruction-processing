# ReconFormer evaluation

Course project for EEE 475/575, Medical Image Reconstruction and Processing.

## Scope

This project evaluates the published, pre-trained ReconFormer model for accelerated MRI reconstruction. The model was not trained here. Evaluation used 30 scans (1,059 slices) from the fastMRI single-coil knee validation data on Google Colab with an NVIDIA T4 GPU.

We ran two acceleration settings:

| Acceleration factor | Center fraction |
| --- | --- |
| 4 | 0.08 |
| 8 | 0.04 |

The reported metrics are PSNR, SSIM, and NMSE. The included result files show quantitative summaries and example reconstructions.

## Evaluation procedure

The notebook uses the upstream ReconFormer implementation and its pre-trained checkpoints. Input files were placed in the directory layout expected by its data loader: `<F_path>/PD/val/*.h5`. The scans, checkpoints, and upstream source code are not stored in this repository.

The recorded evaluation changes were:

- Run on one GPU (`--gpu 0`).
- Update the legacy Fourier transform helpers to use the modern `torch.fft` API while retaining the model's real/imaginary channel format.
- Reduce the test batch size from 8 to 1 to fit the T4 GPU memory.
- Save a four-panel comparison for a fixed scan and slice: reference image, zero-filled input, model output, and absolute error.

These changes were made in the Colab copy of the upstream code; that modified source is not included here. The results therefore document one evaluation run and should not be treated as a turnkey reproduction or a full reproduction of the paper's experiments. The 30-scan subset also limits direct comparison with results reported over a larger dataset.

## Files

- `ReconFormer_Eval.ipynb` — evaluation notebook.
- `quantitativeResults.pdf` — PSNR, SSIM, and NMSE summary.
- `qualitativeResults.pdf` — example image comparisons.
- `ReconFormer_Professor_Ready_Presentation.pptx` — project presentation.

## Generative AI disclosure

An AI assistant was used to help debug PyTorch API incompatibilities, update the Fourier transform calls, and implement qualitative figure saving. The group reviewed and tested the changes and can explain them. Add the tool name and version required by the course when completing the project report's disclosure.
