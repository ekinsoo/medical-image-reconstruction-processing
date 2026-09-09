# ReconFormer Evaluation — Workflow & Code Modifications

**EEE 475/575 Project — Accelerated MRI Reconstruction**

This document explains, step by step, how we ran the pre-trained ReconFormer model on the fastMRI validation data, and which modifications we made to the original repository. It is meant both as an internal reference for the group and as a basis for the *Data and Experimental Setup* and *Appendix* sections of the report.

---

## 1. Goal

We did **not** train the model. Our objective was to take the authors' published pre-trained weights and run **inference (testing) only** on the fastMRI validation set, then compute image-quality metrics (PSNR, SSIM, NMSE) and produce qualitative comparison figures. This matches the undergraduate project requirement: run an existing implementation and evaluate it, rather than reproduce or improve it.

## 2. Computational Environment

We ran everything on **Google Colab** with an **NVIDIA Tesla T4** GPU (15 GB).

The reason we did not use a local machine: the original code was written for **PyTorch 1.7 / Python 3.6 (2020)**. A local RTX 50-series GPU (Blackwell architecture) is not supported by old PyTorch versions and requires a very recent build, which creates a difficult dependency situation. The Colab T4 GPU works cleanly with a wide range of PyTorch versions, so it was the lower-risk choice.

## 3. Step-by-Step Procedure

### Step 1 — Verify the GPU
Confirmed Colab assigned a T4 GPU and that PyTorch detected CUDA. The T4's architecture is compatible with the model, so no GPU-related issues were expected at this stage.

### Step 2 — Clone the repository
Cloned the official ReconFormer repository from GitHub. This gave us the model code, the evaluation script (`main_recon_test.py`), the data loaders, and the shell script `run_recon_eval.sh`.

### Step 3 — Install dependencies
We did **not** use the repository's `conda_environment.yml`, because it is pinned to 2020-era Linux-specific package versions that no longer install correctly. Instead we installed only the packages the evaluation script actually needs:
`h5py` (reading .h5 files), `scikit-image` (PSNR/SSIM/NMSE metrics), `runstats` (statistics used by the fastMRI data loader), `timm` (Transformer building blocks, pinned to version 0.4.12 because the code was written against that API), `opencv-python`, and `tqdm`. PyTorch itself was already provided by Colab and was left untouched.

Packages used only for training or visualization (TensorBoard, PyTorch-Lightning, pandas, seaborn, etc.) were skipped, since the evaluation pipeline does not require them.

### Step 4 — Prepare the dataset
We used the **preprocessed fastMRI single-coil knee dataset** provided by the authors (the data are not modified, only reformatted to match their data loader). We downloaded **only the validation split**, since we are not training. In the ReconFormer setup, the official fastMRI validation set *is* the test set.

To keep download size and runtime manageable, we used a subset of **30 scans**, which together contain **1059 slices**. The data loader expects the directory layout `<F_path>/PD/val/*.h5`, so the files were placed on Google Drive in that structure and Drive was mounted into Colab.

Each `.h5` file is one MRI scan: it contains a `kspace` array (complex-valued raw frequency-domain data, ~35 slices of 320×320) and a `reconstruction_esc` array (the fully-sampled ground-truth image used as the reference for metrics).

### Step 5 — Sanity-check the data
Before running the full evaluation, we inspected one `.h5` file to confirm the keys, array shapes, and data types matched what the model expects. This verified the data downloaded correctly.

### Step 6 — Run the evaluation
We ran `main_recon_test.py` with the arguments from `run_recon_eval.sh`: single-coil challenge, ReconFormer model, the fastMRI dataset path, the PD sequence, the chosen acceleration factor, and the path to the pre-trained checkpoint. We ran the evaluation twice, once for each acceleration factor (AF=4 with center fraction 0.08, AF=8 with center fraction 0.04), using the matching pre-trained checkpoints. Reaching a successful run required four code modifications, described in Section 4.

## 4. Modifications Made to the Original Repository

The original code runs on PyTorch 1.7. We ran it on a modern PyTorch (2.x) provided by Colab. Three issues had to be resolved for the code to run, and one further modification was made deliberately to produce qualitative figures. **None of these change the model architecture or the reconstruction algorithm** — they update deprecated API calls, adjust runtime settings, and add a visualization step.

### Modification 1 — Single-GPU instead of multi-GPU
**File:** `main_recon_test.py` (run argument)
**Problem:** The script defaults to multi-GPU mode (`DataParallel`). On a single-GPU machine like Colab, this caused an "Invalid device id" error.
**Fix:** We passed `--gpu 0` so the script uses exactly one GPU and skips the `DataParallel` branch.
**Why it is safe:** This only selects how many GPUs are used; the computation is identical.

### Modification 2 — Updated the Fourier transform functions
**File:** `data/transforms.py`
**Problem:** The code uses `torch.fft`, `torch.ifft`, `torch.rfft`, and `torch.irfft`. These functions were **removed** from PyTorch in version 1.8. They represented complex numbers as a real array with a final dimension of size 2 (real and imaginary parts).
**Fix:** We rewrote the four functions (`fft2`, `ifft2`, `rfft2`, `irfft2`) to use the modern `torch.fft` module (`torch.fft.fftn` / `torch.fft.ifftn`). Inside each function, the real-valued `[..., 2]` array is converted to a true complex tensor, the transform is applied, and the result is converted back to the `[..., 2]` format. We also added `.contiguous()` because the modern API requires the input to be laid out contiguously in memory.
**Why it is safe:** The input and output formats of these functions are unchanged, so the rest of the code (including the data-consistency layers) works without further modification. This is purely an API update; the mathematical operation — a centered 2D FFT/IFFT — is the same.

### Modification 3 — Reduced the batch size
**File:** `main_recon_test.py` (data loader configuration)
**Problem:** The evaluation script processes 8 slices at once (batch size 8). The ReconFormer attention layers are memory-heavy, and 8 slices exceeded the T4's 15 GB of memory, causing a CUDA out-of-memory error.
**Fix:** We changed the test data loader's batch size from 8 to 1.
**Why it is safe:** Batch size only affects how many images are processed simultaneously, not the result for each image. A smaller batch is slower but produces identical reconstructions and metrics.

### Modification 4 — Added qualitative figure generation
**File:** `models/evaluation.py`, function `test_recon_save`
**Motivation:** The original `test_recon_save` function computes PSNR/SSIM/NMSE and internally stores the reconstructed volumes, ground-truth volumes, and zero-filled inputs in dictionaries, but it never writes any images to disk — it only returns the numeric metrics. The project requires a qualitative evaluation (visual comparison), so we needed reconstructed images, not just numbers.
**Change:** Rather than writing a separate script, we extended `test_recon_save` itself. Just before the function's `return` statement, we added a block that reuses the already-computed `outputs_save`, `targets_save`, and `input_save` dictionaries. For one representative scan, it selects a central slice and saves a four-panel figure: ground truth, zero-filled reconstruction, ReconFormer reconstruction, and the error map (absolute difference between ground truth and reconstruction). The figure filename includes the acceleration factor, so running the evaluation for AF=4 and AF=8 produces two separate figures.
**Why it is appropriate:** This is an addition, not a replacement — the metric computation and the function's return value are completely unchanged. The block only reads data the function had already prepared and writes a visualization. Because it always picks the first scan (sorted by filename) and the central slice, the figures for AF=4 and AF=8 correspond to exactly the same scan and slice, so the two acceleration factors can be compared directly.

## 5. Outcome

After these modifications, the evaluation ran to completion for both acceleration factors. For each AF the model loaded the corresponding pre-trained weights, reconstructed the under-sampled validation slices, reported PSNR, SSIM, and NMSE (computed per volume and averaged over the 30 scans), and saved a qualitative comparison figure.

Any small differences between our numbers and the paper's are attributable to the evaluation subset: the paper averages over 199 scans, whereas we used 30 scans. The project guidelines explicitly state that students are not expected to reproduce all analyses, so a representative subset is sufficient to demonstrate the method's behavior.

## 6. Suggested Wording for the Report

For the *Data and Experimental Setup* / *Appendix* sections:

> We evaluated the pre-trained ReconFormer model on a subset of the fastMRI single-coil knee validation set (30 scans, 1059 slices in total) using the authors' open-source implementation. Experiments were run on Google Colab with an NVIDIA T4 GPU. Because the original code targets PyTorch 1.7, three modifications were required to run it on a current PyTorch version: (i) the evaluation script was set to single-GPU mode, (ii) the deprecated `torch.fft`/`torch.ifft`/`torch.rfft`/`torch.irfft` calls in the Fourier-transform utilities were rewritten using the modern `torch.fft` module while preserving the original input/output conventions, and (iii) the test batch size was reduced from 8 to 1 to fit within GPU memory. In addition, the evaluation function was extended to save four-panel qualitative figures (ground truth, zero-filled input, reconstruction, error map) for a fixed scan and slice. None of these changes affect the model architecture or the reconstruction algorithm. Evaluation parameters (acceleration factor, center fraction, unrolling length) were inherited from the original implementation.

## 7. GenAI Disclosure Note

The course requires disclosing any use of generative AI. We used an AI assistant for help with debugging the PyTorch API incompatibilities, adapting the deprecated FFT functions, and writing the qualitative figure-generation block. The group reviewed, tested, and is able to explain every modification; the AI assistance was limited to identifying the cause of errors and proposing the modified code. This note should be included in the report's GenAI disclosure section, along with the tool name and version.
