clear; close all; clc;
%% Setup
outDir = fullfile(pwd, 'outputs');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

diaryFile = fullfile(outDir, 'console_output.txt');
if exist(diaryFile, 'file')
    try
        delete(diaryFile);
    catch
    end
end
diary(diaryFile);
cleanupObj = onCleanup(@() diary('off'));

if ~exist('@Wavelet', 'dir')
    if exist('@Wavelet.zip', 'file')
        fprintf('Extracting @Wavelet.zip...\n');
        unzip('@Wavelet.zip', pwd);
    else
        error('The @Wavelet folder or @Wavelet.zip file is missing.');
    end
end
addpath(pwd);

dataFile = 'multicoil-random.mat';
if ~exist(dataFile, 'file')
    error('Could not find %s in the current folder.', dataFile);
end
load(dataFile, 'im', 'mask');

im = double(im);
mask = logical(mask);
[Nx, Ny, Nc] = size(im);

betaListPart3 = [1e-4, 1e-3, 1e-2, 1e-1];
lambdaKernelDisplay = 0;
lambdaSpirit = 1e-3;
betaL1Spirit = 1e-4;
Niter = 10;
showFigures = true;   

resultRows = cell(0, 6);

fprintf('EEE 475/575 HW5 l1-SPIRiT reconstruction\n');
fprintf('Student: Ozgur Ekin Sonmez, ID: 22201966\n');
fprintf('Data dimensions: Nx = %d, Ny = %d, Nc = %d\n', Nx, Ny, Nc);
fprintf('Parameters: lambdaSpirit = %.3g, betaL1Spirit = %.3g, Niter = %d\n\n', ...
    lambdaSpirit, betaL1Spirit, Niter);

%% Part 1 - Display the fully sampled data
kfull = fft2cStack(im);

refSosRaw = sosCombine(im);
refSos = normalizeByP95(refSosRaw);
max_val = max(refSos(:));

saveMontageFigure(abs(im), [], fullfile(outDir, 'part1_coil_images.png'), ...
    'Part 1: Fully sampled coil magnitude images');
saveMontageFigure(log1p(abs(kfull)), [], fullfile(outDir, 'part1_kspace_spectra.png'), ...
    'Part 1: Fully sampled k-space spectra');
saveImageFigure(refSos, [0 max_val], fullfile(outDir, 'part1_sos_reference.png'), ...
    'Part 1: SoS reference image');

fprintf('Part 1\n');
fprintf('Reference SoS 95th percentile before normalization: %.6g\n', prctile(refSosRaw(:), 95));
fprintf('max_val after 95th percentile normalization: %.6g\n\n', max_val);

%% Part 2 - Sampling mask and random undersampled images
R = 1 / nnz(mask) * numel(mask);
[imu, Mu] = randundersample(im, mask);

zfSos = normalizeByP95(sosCombine(imu));
[psnrZF, ssimZF] = computeQuality(zfSos, refSos, max_val);
errZF = abs(zfSos - refSos);
resultRows = addMetric(resultRows, 'Part 2 zero-fill', 0, NaN, NaN, psnrZF, ssimZF);

saveImageFigure(double(mask), [0 1], fullfile(outDir, 'part2_mask.png'), ...
    sprintf('Part 2: Random sampling mask, R = %.3f', R));
saveMontageFigure(log1p(abs(Mu)), [], fullfile(outDir, 'part2_kspace_spectra.png'), ...
    'Part 2: Random undersampled k-space spectra');
saveMontageFigure(abs(imu), [], fullfile(outDir, 'part2_zerofill_coil_images.png'), ...
    'Part 2: Zero-fill coil magnitude images');
saveImageFigure(zfSos, [0 max_val], fullfile(outDir, 'part2_zerofill_sos.png'), ...
    'Part 2: Zero-fill SoS image');
saveImageFigure(errZF, [0 0.2*max_val], fullfile(outDir, 'part2_zerofill_error.png'), ...
    'Part 2: Zero-fill absolute error');

fprintf('Part 2\n');
fprintf('Acquired k-space samples: %d / %d\n', nnz(mask), numel(mask));
fprintf('Overall acceleration R: %.6f\n', R);
fprintf('Zero-fill PSNR: %.4f dB\n', psnrZF);
fprintf('Zero-fill SSIM: %.6f\n\n', ssimZF);

%% Part 3 - l1 regularization in the wavelet domain
coeffBefore = zeros(Nx, Ny, Nc);
for c = 1:Nc
    wv = Wavelet('Daubechies', 4, 4);
    coeffBefore(:, :, c) = wv * imu(:, :, c);
end
saveMontageFigure(log1p(abs(coeffBefore)), [], ...
    fullfile(outDir, 'part3_wavelet_coeff_before.png'), ...
    'Part 3: Wavelet coefficient magnitudes before thresholding');

fprintf('Part 3\n');
fprintf('Wavelet beta sweep:\n');
psnrWavelet = zeros(numel(betaListPart3), 1);
ssimWavelet = zeros(numel(betaListPart3), 1);
for b = 1:numel(betaListPart3)
    beta = betaListPart3(b);
    imth = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        imth(:, :, c) = l1wavelet(imu(:, :, c), beta);
    end

    thSos = normalizeByP95(sosCombine(imth));
    [psnrTh, ssimTh] = computeQuality(thSos, refSos, max_val);
    psnrWavelet(b) = psnrTh;
    ssimWavelet(b) = ssimTh;
    errTh = abs(thSos - refSos);
    betaTag = numberTag(beta);
    resultRows = addMetric(resultRows, sprintf('Part 3 beta %g', beta), 0, NaN, beta, psnrTh, ssimTh);

    saveImageFigure(thSos, [0 max_val], ...
        fullfile(outDir, ['part3_beta_' betaTag '_sos.png']), ...
        sprintf('Part 3: Wavelet l1 SoS, beta = %g', beta));
    saveImageFigure(errTh, [0 0.2*max_val], ...
        fullfile(outDir, ['part3_beta_' betaTag '_error.png']), ...
        sprintf('Part 3: Wavelet l1 error, beta = %g', beta));

    fprintf('  beta = %.3g: PSNR = %.4f dB, SSIM = %.6f\n', beta, psnrTh, ssimTh);
end
fprintf('\n');

%% Part 4 - SPIRiT kernel calibration
Mc = extractCalibration(kfull, [32 32]);
kernel0 = calibrateSpirit(Mc, lambdaKernelDisplay);
kernel = calibrateSpirit(Mc, lambdaSpirit);

saveKernelFigure(kernel0, fullfile(outDir, 'part4_kernel_lambda0.png'), ...
    'Part 4: SPIRiT kernel magnitude, lambda = 0');

fprintf('Part 4\n');
fprintf('Calibration region size: %d x %d x %d\n', size(Mc, 1), size(Mc, 2), size(Mc, 3));
fprintf('Kernel size: %d x %d\n', size(kernel, 1), size(kernel, 2));
fprintf('Displayed kernel was calibrated with lambda = %.3g\n', lambdaKernelDisplay);
fprintf('Reconstruction kernel was calibrated with lambda = %.3g\n\n', lambdaSpirit);

%% Part 5 - One SPIRiT iteration without data consistency
[imSpirit1, MSpirit1] = spirit(Mu, kernel);
spirit1Sos = normalizeByP95(sosCombine(imSpirit1));
[psnrSpirit1, ssimSpirit1] = computeQuality(spirit1Sos, refSos, max_val);
errSpirit1 = abs(spirit1Sos - refSos);
resultRows = addMetric(resultRows, 'Part 5 SPIRiT 1 no DC', 1, lambdaSpirit, NaN, psnrSpirit1, ssimSpirit1);

saveMontageFigure(abs(imSpirit1), [], fullfile(outDir, 'part5_spirit_coils.png'), ...
    'Part 5: One SPIRiT iteration coil images');
saveMontageFigure(log1p(abs(MSpirit1)), [], fullfile(outDir, 'part5_spirit_kspace.png'), ...
    'Part 5: One SPIRiT iteration k-space spectra');
saveImageFigure(spirit1Sos, [0 max_val], fullfile(outDir, 'part5_spirit_sos.png'), ...
    'Part 5: One SPIRiT iteration SoS image');
saveImageFigure(errSpirit1, [0 0.2*max_val], fullfile(outDir, 'part5_spirit_error.png'), ...
    'Part 5: One SPIRiT iteration absolute error');

fprintf('Part 5\n');
fprintf('One SPIRiT iteration without data consistency: PSNR = %.4f dB, SSIM = %.6f\n\n', ...
    psnrSpirit1, ssimSpirit1);

%% Part 6 - One full SPIRiT iteration with data consistency
MFull1 = applyDataConsistency(MSpirit1, Mu, mask);
imFull1 = ifft2cStack(MFull1);
full1Sos = normalizeByP95(sosCombine(imFull1));
[psnrFull1, ssimFull1] = computeQuality(full1Sos, refSos, max_val);
errFull1 = abs(full1Sos - refSos);
resultRows = addMetric(resultRows, 'Part 6 SPIRiT 1 with DC', 1, lambdaSpirit, NaN, psnrFull1, ssimFull1);

saveMontageFigure(abs(imFull1), [], fullfile(outDir, 'part6_full1_coils.png'), ...
    'Part 6: One full SPIRiT iteration coil images');
saveMontageFigure(log1p(abs(MFull1)), [], fullfile(outDir, 'part6_full1_kspace.png'), ...
    'Part 6: One full SPIRiT iteration k-space spectra');
saveImageFigure(full1Sos, [0 max_val], fullfile(outDir, 'part6_full1_sos.png'), ...
    'Part 6: One full SPIRiT iteration SoS image');
saveImageFigure(errFull1, [0 0.2*max_val], fullfile(outDir, 'part6_full1_error.png'), ...
    'Part 6: One full SPIRiT iteration absolute error');

fprintf('Part 6\n');
fprintf('One SPIRiT iteration with data consistency: PSNR = %.4f dB, SSIM = %.6f\n\n', ...
    psnrFull1, ssimFull1);

%% Part 7 - Ten full SPIRiT iterations with data consistency
psnrSpirit = zeros(Niter, 1);
ssimSpirit = zeros(Niter, 1);
MrIter = Mu;
imIter = imu;

for it = 1:Niter
    [~, MrTmp] = spirit(MrIter, kernel);
    MrIter = applyDataConsistency(MrTmp, Mu, mask);
    imIter = ifft2cStack(MrIter);

    iterSos = normalizeByP95(sosCombine(imIter));
    [psnrSpirit(it), ssimSpirit(it)] = computeQuality(iterSos, refSos, max_val);
    resultRows = addMetric(resultRows, 'Part 7 SPIRiT iterations', it, lambdaSpirit, NaN, ...
        psnrSpirit(it), ssimSpirit(it));
end

errSpiritLast = abs(iterSos - refSos);
saveMetricPlot(psnrSpirit, 'PSNR (dB)', fullfile(outDir, 'part7_psnr_iterations.png'), ...
    'Part 7: SPIRiT PSNR versus iteration');
saveMetricPlot(ssimSpirit, 'SSIM', fullfile(outDir, 'part7_ssim_iterations.png'), ...
    'Part 7: SPIRiT SSIM versus iteration');
saveMontageFigure(abs(imIter), [], fullfile(outDir, 'part7_last_coils.png'), ...
    'Part 7: Last SPIRiT iteration coil images');
saveMontageFigure(log1p(abs(MrIter)), [], fullfile(outDir, 'part7_last_kspace.png'), ...
    'Part 7: Last SPIRiT iteration k-space spectra');
saveImageFigure(iterSos, [0 max_val], fullfile(outDir, 'part7_last_sos.png'), ...
    'Part 7: Last SPIRiT iteration SoS image');
saveImageFigure(errSpiritLast, [0 0.2*max_val], fullfile(outDir, 'part7_last_error.png'), ...
    'Part 7: Last SPIRiT iteration absolute error');

fprintf('Part 7\n');
fprintf('Ten SPIRiT iterations with data consistency:\n');
for it = 1:Niter
    fprintf('  iter %2d: PSNR = %.4f dB, SSIM = %.6f\n', it, psnrSpirit(it), ssimSpirit(it));
end
fprintf('\n');

%% Part 8 - One iteration of l1-SPIRiT reconstruction
[imL1OneAll, ML1OneAll] = l1spirit(Mu, mask, kernel, betaL1Spirit, 1);
imL1One = imL1OneAll(:, :, :, 1);
ML1One = ML1OneAll(:, :, :, 1);
l1OneSos = normalizeByP95(sosCombine(imL1One));
[psnrL1One, ssimL1One] = computeQuality(l1OneSos, refSos, max_val);
errL1One = abs(l1OneSos - refSos);
resultRows = addMetric(resultRows, 'Part 8 l1-SPIRiT 1', 1, lambdaSpirit, betaL1Spirit, ...
    psnrL1One, ssimL1One);

saveMontageFigure(abs(imL1One), [], fullfile(outDir, 'part8_l1spirit1_coils.png'), ...
    'Part 8: One l1-SPIRiT iteration coil images');
saveMontageFigure(log1p(abs(ML1One)), [], fullfile(outDir, 'part8_l1spirit1_kspace.png'), ...
    'Part 8: One l1-SPIRiT iteration k-space spectra');
saveImageFigure(l1OneSos, [0 max_val], fullfile(outDir, 'part8_l1spirit1_sos.png'), ...
    'Part 8: One l1-SPIRiT iteration SoS image');
saveImageFigure(errL1One, [0 0.2*max_val], fullfile(outDir, 'part8_l1spirit1_error.png'), ...
    'Part 8: One l1-SPIRiT iteration absolute error');

fprintf('Part 8\n');
fprintf('One l1-SPIRiT iteration: beta = %.3g, PSNR = %.4f dB, SSIM = %.6f\n\n', ...
    betaL1Spirit, psnrL1One, ssimL1One);

%% Part 9 - Ten iterations of l1-SPIRiT reconstruction
[imL1All, ML1All] = l1spirit(Mu, mask, kernel, betaL1Spirit, Niter);
psnrL1 = zeros(Niter, 1);
ssimL1 = zeros(Niter, 1);

for it = 1:Niter
    l1SosIt = normalizeByP95(sosCombine(imL1All(:, :, :, it)));
    [psnrL1(it), ssimL1(it)] = computeQuality(l1SosIt, refSos, max_val);
    resultRows = addMetric(resultRows, 'Part 9 l1-SPIRiT iterations', it, lambdaSpirit, betaL1Spirit, ...
        psnrL1(it), ssimL1(it));
end

l1LastSos = normalizeByP95(sosCombine(imL1All(:, :, :, Niter)));
errL1Last = abs(l1LastSos - refSos);

saveMetricPlot(psnrL1, 'PSNR (dB)', fullfile(outDir, 'part9_psnr_iterations.png'), ...
    'Part 9: l1-SPIRiT PSNR versus iteration');
saveMetricPlot(ssimL1, 'SSIM', fullfile(outDir, 'part9_ssim_iterations.png'), ...
    'Part 9: l1-SPIRiT SSIM versus iteration');
saveImageFigure(l1LastSos, [0 max_val], fullfile(outDir, 'part9_last_sos.png'), ...
    'Part 9: Last l1-SPIRiT iteration SoS image');
saveImageFigure(errL1Last, [0 0.2*max_val], fullfile(outDir, 'part9_last_error.png'), ...
    'Part 9: Last l1-SPIRiT iteration absolute error');

fprintf('Part 9\n');
fprintf('Ten l1-SPIRiT iterations:\n');
for it = 1:Niter
    fprintf('  iter %2d: PSNR = %.4f dB, SSIM = %.6f\n', it, psnrL1(it), ssimL1(it));
end
fprintf('\n');

%% Save numerical outputs for the LaTeX report
metricsTable = cell2table(resultRows, ...
    'VariableNames', {'Method', 'Iteration', 'Lambda', 'Beta', 'PSNR_dB', 'SSIM'});
writeMetricsCsv(metricsTable, fullfile(outDir, 'results_table.csv'));
writeLatexMacros(fullfile(outDir, 'results_macros.tex'), R, max_val, lambdaSpirit, ...
    betaL1Spirit, betaListPart3, psnrWavelet, ssimWavelet, psnrZF, ssimZF, ...
    psnrSpirit1, ssimSpirit1, psnrFull1, ssimFull1, psnrSpirit, ssimSpirit, ...
    psnrL1One, ssimL1One, psnrL1, ssimL1);

save(fullfile(outDir, 'hw5_results.mat'), 'R', 'max_val', 'lambdaSpirit', 'betaL1Spirit', ...
    'betaListPart3', 'psnrWavelet', 'ssimWavelet', 'psnrZF', 'ssimZF', 'psnrSpirit1', 'ssimSpirit1', ...
    'psnrFull1', 'ssimFull1', 'psnrSpirit', 'ssimSpirit', 'psnrL1One', ...
    'ssimL1One', 'psnrL1', 'ssimL1', 'metricsTable');

fprintf('Saved figures and outputs to: %s\n', outDir);

%% Local functions
function [imu, Mu] = randundersample(im, mask)
    [Nx, Ny, Nc] = size(im);
    Mu = zeros(Nx, Ny, Nc);
    imu = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        Mc = fft2c(im(:, :, c));
        Mu(:, :, c) = Mc .* mask;
        imu(:, :, c) = ifft2c(Mu(:, :, c));
    end
end

function imth = l1wavelet(imr, beta)
    wv = Wavelet('Daubechies', 4, 4);
    coeffW = wv * imr;
    maxCoeff = max(abs(coeffW(:)));
    if maxCoeff == 0
        imth = imr;
        return;
    end
    mag = abs(coeffW);
    shrink = max(0, mag - beta * maxCoeff) ./ (mag + eps);
    coeffWth = coeffW .* shrink;
    imth = wv' * coeffWth;
end

function kernel = calibrateSpirit(Mc, lambda)
    [Nx, Ny, Nc] = size(Mc);
    offsets = -1:1;
    nFeat = 9 * Nc - 1;
    nRows = (Nx - 2) * (Ny - 2);
    kernel = zeros(nFeat, Nc);

    for targetCoil = 1:Nc
        A = zeros(nRows, nFeat);
        y = zeros(nRows, 1);
        row = 0;
        for ix = 2:Nx-1
            for iy = 2:Ny-1
                row = row + 1;
                feat = 0;
                for coil = 1:Nc
                    for dx = offsets
                        for dy = offsets
                            if coil == targetCoil && dx == 0 && dy == 0
                                continue;
                            end
                            feat = feat + 1;
                            A(row, feat) = Mc(ix + dx, iy + dy, coil);
                        end
                    end
                end
                y(row) = Mc(ix, iy, targetCoil);
            end
        end

        normalMat = A' * A;
        if lambda > 0
            regScale = trace(normalMat) / nFeat;
            normalMat = normalMat + lambda * regScale * eye(nFeat);
        end
        kernel(:, targetCoil) = normalMat \ (A' * y);
    end
end

function [imr, Mr] = spirit(Mu, kernel)
    [Nx, Ny, Nc] = size(Mu);
    offsets = -1:1;
    Mr = zeros(Nx, Ny, Nc);

    for targetCoil = 1:Nc
        feat = 0;
        estimate = zeros(Nx, Ny);
        for coil = 1:Nc
            for dx = offsets
                for dy = offsets
                    if coil == targetCoil && dx == 0 && dy == 0
                        continue;
                    end
                    feat = feat + 1;
                    neighbor = shift2dZero(Mu(:, :, coil), dx, dy);
                    estimate = estimate + kernel(feat, targetCoil) * neighbor;
                end
            end
        end
        Mr(:, :, targetCoil) = estimate;
    end

    imr = ifft2cStack(Mr);
end

function [imr, Mr] = l1spirit(Mu, mask, kernel, beta, Niter)
    [Nx, Ny, Nc] = size(Mu);
    Mr = zeros(Nx, Ny, Nc, Niter);
    imr = zeros(Nx, Ny, Nc, Niter);
    MrCur = Mu;

    for it = 1:Niter
        MrCur = applyDataConsistency(MrCur, Mu, mask);
        [imSpirit, ~] = spirit(MrCur, kernel);

        imTh = zeros(Nx, Ny, Nc);
        for c = 1:Nc
            imTh(:, :, c) = l1wavelet(imSpirit(:, :, c), beta);
        end

        MrCur = fft2cStack(imTh);
        imr(:, :, :, it) = imTh;
        Mr(:, :, :, it) = MrCur;
    end
end

function Mdc = applyDataConsistency(Mr, Mu, mask)
    Mdc = Mr;
    for c = 1:size(Mr, 3)
        tmp = Mdc(:, :, c);
        acquired = Mu(:, :, c);
        tmp(mask) = acquired(mask);
        Mdc(:, :, c) = tmp;
    end
end

function Mc = extractCalibration(M, calSize)
    [Nx, Ny, ~] = size(M);
    cx = floor(Nx/2) + 1;
    cy = floor(Ny/2) + 1;
    hx = calSize(1) / 2;
    hy = calSize(2) / 2;
    Mc = M(cx-hx:cx+hx-1, cy-hy:cy+hy-1, :);
end

function M = fft2cStack(im)
    M = zeros(size(im));
    for c = 1:size(im, 3)
        M(:, :, c) = fft2c(im(:, :, c));
    end
end

function im = ifft2cStack(M)
    im = zeros(size(M));
    for c = 1:size(M, 3)
        im(:, :, c) = ifft2c(M(:, :, c));
    end
end

function M = fft2c(im)
    M = fftshift(fft2(ifftshift(im)));
end

function im = ifft2c(M)
    im = fftshift(ifft2(ifftshift(M)));
end

function out = shift2dZero(in, dx, dy)
    [Nx, Ny] = size(in);
    out = zeros(Nx, Ny);

    dstX = max(1, 1 - dx):min(Nx, Nx - dx);
    dstY = max(1, 1 - dy):min(Ny, Ny - dy);
    srcX = dstX + dx;
    srcY = dstY + dy;

    out(dstX, dstY) = in(srcX, srcY);
end

function sos = sosCombine(im)
    sos = sqrt(sum(abs(im).^2, 3));
end

function imn = normalizeByP95(im)
    p95 = prctile(abs(im(:)), 95);
    if p95 == 0
        imn = abs(im);
    else
        imn = abs(im) / p95;
    end
end

function [p, s] = computeQuality(imTest, imRef, maxVal)
    testMetric = double(imTest) / maxVal;
    refMetric = double(imRef) / maxVal;
    mseVal = mean((testMetric(:) - refMetric(:)).^2);
    if mseVal == 0
        p = Inf;
    else
        p = 10 * log10(1 / mseVal);
    end

    if exist('ssim', 'file') == 2 || exist('ssim', 'builtin') == 5
        s = ssim(testMetric, refMetric, 'DynamicRange', 1);
    else
        s = simpleSSIM(testMetric, refMetric);
    end
end

function s = simpleSSIM(x, y)
    x = x(:);
    y = y(:);
    C1 = 0.01^2;
    C2 = 0.03^2;
    mux = mean(x);
    muy = mean(y);
    sigx = var(x, 1);
    sigy = var(y, 1);
    sigxy = mean((x - mux) .* (y - muy));
    s = ((2*mux*muy + C1) * (2*sigxy + C2)) / ...
        ((mux^2 + muy^2 + C1) * (sigx + sigy + C2));
end

function saveMontageFigure(stack, displayRange, filename, figTitle)
    vol = reshapeForMontage(stack);
    f = figure('Color', 'w', 'Visible', figureVisibility());
    if isempty(displayRange)
        montage(vol, 'DisplayRange', []);
    else
        montage(vol, 'DisplayRange', displayRange);
    end
    colormap gray; title(figTitle, 'Interpreter', 'none');
    exportgraphics(f, filename, 'Resolution', 300);
    finalizeFigure(f);
end

function saveImageFigure(im, displayRange, filename, figTitle)
    f = figure('Color', 'w', 'Visible', figureVisibility());
    imshow(im, displayRange); colormap gray; axis image off;
    title(figTitle, 'Interpreter', 'none');
    exportgraphics(f, filename, 'Resolution', 300);
    finalizeFigure(f);
end

function saveKernelFigure(kernel, filename, figTitle)
    f = figure('Color', 'w', 'Visible', figureVisibility());
    imagesc(abs(kernel)); axis image; colorbar; colormap parula;
    xlabel('Target coil'); ylabel('Kernel coefficient');
    title(figTitle, 'Interpreter', 'none');
    exportgraphics(f, filename, 'Resolution', 300);
    finalizeFigure(f);
end

function saveMetricPlot(values, yLabelText, filename, figTitle)
    f = figure('Color', 'w', 'Visible', figureVisibility());
    plot(1:numel(values), values, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
    grid on; xlabel('Iteration'); ylabel(yLabelText);
    title(figTitle, 'Interpreter', 'none');
    xlim([1 numel(values)]);
    exportgraphics(f, filename, 'Resolution', 300);
    finalizeFigure(f);
end

function vis = figureVisibility()
    if evalin('base', 'exist(''showFigures'', ''var'')')
        showFigures = evalin('base', 'showFigures');
    else
        showFigures = true;
    end
    if showFigures
        vis = 'on';
    else
        vis = 'off';
    end
end

function finalizeFigure(f)
    if strcmp(figureVisibility(), 'on')
        drawnow;
    else
        close(f);
    end
end

function vol = reshapeForMontage(stack)
    if ndims(stack) == 2
        vol = stack;
    else
        vol = permute(stack, [1 2 4 3]);
    end
end

function tag = numberTag(x)
    tag = sprintf('%.0e', x);
    tag = strrep(tag, '-', 'm');
    tag = strrep(tag, '+', '');
    tag = strrep(tag, '.', 'p');
end

function rows = addMetric(rows, method, iter, lambda, beta, psnrVal, ssimVal)
    rows(end+1, :) = {method, iter, lambda, beta, psnrVal, ssimVal};
end

function writeMetricsCsv(T, filename)
    try
        writetable(T, filename);
    catch
        fid = fopen(filename, 'w');
        fprintf(fid, 'Method,Iteration,Lambda,Beta,PSNR_dB,SSIM\n');
        for i = 1:height(T)
            fprintf(fid, '%s,%g,%g,%g,%g,%g\n', T.Method{i}, T.Iteration(i), ...
                T.Lambda(i), T.Beta(i), T.PSNR_dB(i), T.SSIM(i));
        end
        fclose(fid);
    end
end

function writeLatexMacros(filename, R, maxVal, lambdaSpirit, betaL1Spirit, ...
    betaListPart3, psnrWavelet, ssimWavelet, psnrZF, ssimZF, psnrSpirit1, ...
    ssimSpirit1, psnrFull1, ssimFull1, psnrSpirit, ssimSpirit, psnrL1One, ...
    ssimL1One, psnrL1, ssimL1)
    fid = fopen(filename, 'w');
    fprintf(fid, '%% Auto-generated by hw5_l1spirit.m\n');
    fprintf(fid, '\\newcommand{\\AccelerationR}{%.3f}\n', R);
    fprintf(fid, '\\newcommand{\\MaxVal}{%.4f}\n', maxVal);
    fprintf(fid, '\\newcommand{\\LambdaSpirit}{%.3g}\n', lambdaSpirit);
    fprintf(fid, '\\newcommand{\\BetaLoneSpirit}{%.3g}\n', betaL1Spirit);
    betaLabels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    for i = 1:numel(betaListPart3)
        label = betaLabels(i);
        fprintf(fid, '\\newcommand{\\WaveletBeta%c}{%.3g}\n', label, betaListPart3(i));
        fprintf(fid, '\\newcommand{\\PSNRWavelet%c}{%.2f}\n', label, psnrWavelet(i));
        fprintf(fid, '\\newcommand{\\SSIMWavelet%c}{%.4f}\n', label, ssimWavelet(i));
    end
    fprintf(fid, '\\newcommand{\\PSNRZF}{%.2f}\n', psnrZF);
    fprintf(fid, '\\newcommand{\\SSIMZF}{%.4f}\n', ssimZF);
    fprintf(fid, '\\newcommand{\\PSNRSpiritOneNoDC}{%.2f}\n', psnrSpirit1);
    fprintf(fid, '\\newcommand{\\SSIMSpiritOneNoDC}{%.4f}\n', ssimSpirit1);
    fprintf(fid, '\\newcommand{\\PSNRSpiritOneDC}{%.2f}\n', psnrFull1);
    fprintf(fid, '\\newcommand{\\SSIMSpiritOneDC}{%.4f}\n', ssimFull1);
    fprintf(fid, '\\newcommand{\\PSNRSpiritLast}{%.2f}\n', psnrSpirit(end));
    fprintf(fid, '\\newcommand{\\SSIMSpiritLast}{%.4f}\n', ssimSpirit(end));
    fprintf(fid, '\\newcommand{\\PSNRLoneSpiritOne}{%.2f}\n', psnrL1One);
    fprintf(fid, '\\newcommand{\\SSIMLoneSpiritOne}{%.4f}\n', ssimL1One);
    fprintf(fid, '\\newcommand{\\PSNRLoneSpiritLast}{%.2f}\n', psnrL1(end));
    fprintf(fid, '\\newcommand{\\SSIMLoneSpiritLast}{%.4f}\n', ssimL1(end));
    fclose(fid);
end
