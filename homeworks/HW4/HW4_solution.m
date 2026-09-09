%% EEE 475/575 Homework 4 - setup
clear; close all; clc;

dataFile = 'multicoil-data.mat';
load(dataFile, 'im', 'map1', 'map2');

figDir = fullfile(pwd, 'figures');
ensure_dir(figDir);

ref_olc_complex = olc_recon(im, map1);
ref_sense = normalize95(abs(ref_olc_complex));
max_val_sense = max(ref_sense(:));

ref_grappa = normalize95(sos_combine(im));
max_val_grappa = max(ref_grappa(:));

sense_cases = [1 2; 1 4; 2 2];

fprintf('Loaded %s: im/map1/map2 = %s\n', dataFile, mat2str(size(im)));
fprintf('Part I-II reference max_val = %.6g\n', max_val_sense);
fprintf('Part III reference max_val = %.6g\n', max_val_grappa);

%% Part 1.1 - display coil images, k-space spectra, and coil sensitivities
M = fft2c(im);

show_montage(abs(im), [], 'Part 1.1: coil image magnitude', ...
    fullfile(figDir, 'part1_1_coil_image_magnitude.png'));
show_montage(angle(im), [-pi pi], 'Part 1.1: coil image phase', ...
    fullfile(figDir, 'part1_1_coil_image_phase.png'));
show_montage(kspace_spectrum(M), [], 'Part 1.1: coil k-space spectra', ...
    fullfile(figDir, 'part1_1_coil_kspace_spectrum.png'));

show_montage(abs(map1), [], 'Part 1.1: map1 sensitivity magnitude', ...
    fullfile(figDir, 'part1_1_map1_magnitude.png'));
show_montage(angle(map1), [-pi pi], 'Part 1.1: map1 sensitivity phase', ...
    fullfile(figDir, 'part1_1_map1_phase.png'));

show_montage(abs(map2), [], 'Part 1.1: map2 sensitivity magnitude', ...
    fullfile(figDir, 'part1_1_map2_magnitude.png'));
show_montage(angle(map2), [-pi pi], 'Part 1.1: map2 sensitivity phase', ...
    fullfile(figDir, 'part1_1_map2_phase.png'));

map_diff = abs(map2 - map1);
show_montage(map_diff, [], 'Part 1.1: absolute sensitivity difference |map2-map1|', ...
    fullfile(figDir, 'part1_1_map_difference.png'));
fprintf('Part 1.1: mean |map2-map1| = %.6g\n', mean(map_diff(:)));

%% Part 1.2 - OLC reconstruction using map1
olc_map1 = normalize95(abs(olc_recon(im, map1)));
max_val_sense = max(olc_map1(:));
ref_sense = olc_map1;

show_single_image(olc_map1, [0 max_val_sense], ...
    'Part 1.2: OLC magnitude using map1', ...
    fullfile(figDir, 'part1_2_olc_map1.png'));

fprintf('Part 1.2: max_val for Parts I-II = %.6g\n', max_val_sense);

%% Part 1.3 - OLC reconstruction using map2
olc_map2 = normalize95(abs(olc_recon(im, map2)));
metrics_13 = compute_metrics(olc_map2, ref_sense, max_val_sense);

show_image_and_error(olc_map2, ref_sense, max_val_sense, ...
    'Part 1.3: OLC using map2', ...
    fullfile(figDir, 'part1_3_olc_map2_and_error.png'));

print_metrics('Part 1.3 OLC map2', metrics_13);

%% Part 1.4 - SoS multi-coil reconstruction
sos_part1 = normalize95(sos_combine(im));
metrics_14 = compute_metrics(sos_part1, ref_sense, max_val_sense);

show_image_and_error(sos_part1, ref_sense, max_val_sense, ...
    'Part 1.4: SoS reconstruction', ...
    fullfile(figDir, 'part1_4_sos_and_error.png'));

print_metrics('Part 1.4 SoS', metrics_14);

%% Part 2.1 - generate undersampled aliased images
undersampled_tests = struct();
for caseIdx = 1:size(sense_cases, 1)
    Rx = sense_cases(caseIdx, 1);
    Ry = sense_cases(caseIdx, 2);
    [imu, Mu] = undersample(im, Rx, Ry);

    fieldName = sprintf('Rx%d_Ry%d', Rx, Ry);
    undersampled_tests.(fieldName).imu = imu;
    undersampled_tests.(fieldName).Mu = Mu;

    show_montage(abs(imu), [], sprintf('Part 2.1: aliased images, Rx=%d, Ry=%d', Rx, Ry), ...
        fullfile(figDir, sprintf('part2_1_images_Rx%d_Ry%d.png', Rx, Ry)));
    show_montage(kspace_spectrum(Mu), [], sprintf('Part 2.1: undersampled k-space, Rx=%d, Ry=%d', Rx, Ry), ...
        fullfile(figDir, sprintf('part2_1_kspace_Rx%d_Ry%d.png', Rx, Ry)));
end

%% Part 2.2 - SENSE reconstruction sanity check, Rx=Ry=1 and lambda=0
[imu_11, ~] = undersample(im, 1, 1);
im_sense_11 = l2sense(imu_11, map1, 1, 1, 0);
metrics_22 = compute_metrics(im_sense_11, ref_sense, max_val_sense);

show_image_and_error(im_sense_11, ref_sense, max_val_sense, ...
    'Part 2.2: SENSE sanity check, Rx=1, Ry=1, lambda=0', ...
    fullfile(figDir, 'part2_2_sense_Rx1_Ry1_lambda0.png'));

print_metrics('Part 2.2 SENSE Rx=1 Ry=1 lambda=0', metrics_22);

%% Part 2.3 - unregularized SENSE for accelerated cases
sense_unreg = struct();
for caseIdx = 1:size(sense_cases, 1)
    Rx = sense_cases(caseIdx, 1);
    Ry = sense_cases(caseIdx, 2);
    [imu, ~] = undersample(im, Rx, Ry);

    im_sense = l2sense(imu, map1, Rx, Ry, 0);
    metrics = compute_metrics(im_sense, ref_sense, max_val_sense);

    fieldName = sprintf('Rx%d_Ry%d', Rx, Ry);
    sense_unreg.(fieldName).image = im_sense;
    sense_unreg.(fieldName).metrics = metrics;

    show_image_and_error(im_sense, ref_sense, max_val_sense, ...
        sprintf('Part 2.3: SENSE Rx=%d, Ry=%d, lambda=0', Rx, Ry), ...
        fullfile(figDir, sprintf('part2_3_sense_Rx%d_Ry%d_lambda0.png', Rx, Ry)));

    print_metrics(sprintf('Part 2.3 SENSE Rx=%d Ry=%d lambda=0', Rx, Ry), metrics);
end

%% Part 2.4 - choose regularization weights for SENSE
lambda_candidates = [0, 10.^(-8:4)];
lambda_sense = zeros(1, size(sense_cases, 1));
sense_reg = struct();
lambda_scores = struct();

for caseIdx = 1:size(sense_cases, 1)
    Rx = sense_cases(caseIdx, 1);
    Ry = sense_cases(caseIdx, 2);
    [imu, ~] = undersample(im, Rx, Ry);

    candidateImages = cell(numel(lambda_candidates), 1);
    scoreRows = zeros(numel(lambda_candidates), 4);

    for k = 1:numel(lambda_candidates)
        lambda = lambda_candidates(k);
        candidate = l2sense(imu, map1, Rx, Ry, lambda);
        metrics = compute_metrics(candidate, ref_sense, max_val_sense);

        candidateImages{k} = candidate;
        scoreRows(k, 1:3) = [lambda, metrics.psnr, metrics.ssim];
    end

    psnrVals = scoreRows(:, 2);
    ssimVals = scoreRows(:, 3);
    psnrScore = (psnrVals - min(psnrVals)) ./ (max(psnrVals) - min(psnrVals) + eps);
    ssimScore = (ssimVals - min(ssimVals)) ./ (max(ssimVals) - min(ssimVals) + eps);
    scoreRows(:, 4) = 0.5 * psnrScore + 0.5 * ssimScore;

    [~, bestIdx] = max(scoreRows(:, 4));
    fieldName = sprintf('Rx%d_Ry%d', Rx, Ry);
    bestLambda = scoreRows(bestIdx, 1);
    bestImage = candidateImages{bestIdx};
    bestMetrics = struct('psnr', scoreRows(bestIdx, 2), 'ssim', scoreRows(bestIdx, 3));

    lambda_sense(caseIdx) = bestLambda;
    sense_reg.(fieldName).image = bestImage;
    sense_reg.(fieldName).lambda = bestLambda;
    sense_reg.(fieldName).metrics = bestMetrics;
    sense_reg.(fieldName).scores = scoreRows;
    lambda_scores.(fieldName) = array2table(scoreRows, ...
        'VariableNames', {'lambda', 'PSNR_dB', 'SSIM', 'combined_score'});

    show_image_and_error(bestImage, ref_sense, max_val_sense, ...
        sprintf('Part 2.4: regularized SENSE Rx=%d, Ry=%d, lambda=%g', Rx, Ry, bestLambda), ...
        fullfile(figDir, sprintf('part2_4_sense_Rx%d_Ry%d_selected.png', Rx, Ry)));

    fprintf('\nPart 2.4 lambda search table for Rx=%d Ry=%d:\n', Rx, Ry);
    disp(lambda_scores.(fieldName));
    fprintf('Part 2.4 selection criterion: highest combined_score.\n');
    fprintf('Part 2.4 Rx=%d Ry=%d selected lambda = %g\n', Rx, Ry, bestLambda);
    print_metrics(sprintf('Part 2.4 SENSE Rx=%d Ry=%d', Rx, Ry), bestMetrics);
end

%% Part 2.5 - SENSE using inaccurate sensitivity maps map2
sense_map2 = struct();
for caseIdx = 1:size(sense_cases, 1)
    Rx = sense_cases(caseIdx, 1);
    Ry = sense_cases(caseIdx, 2);
    lambda = lambda_sense(caseIdx);
    [imu, ~] = undersample(im, Rx, Ry);

    im_sense = l2sense(imu, map2, Rx, Ry, lambda);
    metrics = compute_metrics(im_sense, ref_sense, max_val_sense);

    fieldName = sprintf('Rx%d_Ry%d', Rx, Ry);
    sense_map2.(fieldName).image = im_sense;
    sense_map2.(fieldName).metrics = metrics;

    show_image_and_error(im_sense, ref_sense, max_val_sense, ...
        sprintf('Part 2.5: SENSE with map2 Rx=%d, Ry=%d, lambda=%g', Rx, Ry, lambda), ...
        fullfile(figDir, sprintf('part2_5_sense_map2_Rx%d_Ry%d.png', Rx, Ry)));

    print_metrics(sprintf('Part 2.5 SENSE map2 Rx=%d Ry=%d', Rx, Ry), metrics);
end

%% Part 2.7 - g-factor maps for selected regularization weights
g_selected = struct();
g_selected_stats = struct();
for caseIdx = 1:size(sense_cases, 1)
    Rx = sense_cases(caseIdx, 1);
    Ry = sense_cases(caseIdx, 2);
    lambda = lambda_sense(caseIdx);

    g = gfactor(map1, Rx, Ry, lambda);
    fieldName = sprintf('Rx%d_Ry%d', Rx, Ry);
    g_selected.(fieldName) = g;
    g_selected_stats.(fieldName) = gfactor_stats(g, map1);

    show_single_image(g, [0 5], ...
        sprintf('Part 2.7: g-factor Rx=%d, Ry=%d, lambda=%g', Rx, Ry, lambda), ...
        fullfile(figDir, sprintf('part2_7_gfactor_Rx%d_Ry%d.png', Rx, Ry)));

    fprintf('Part 2.7 g-factor Rx=%d Ry=%d lambda=%g: max = %.4f, mean = %.4f\n', ...
        Rx, Ry, lambda, g_selected_stats.(fieldName).max, g_selected_stats.(fieldName).mean);
end

%% Part 2.8 - g-factor maps with insufficient regularization
g_insufficient = struct();
g_insufficient_stats = struct();
for caseIdx = 1:size(sense_cases, 1)
    Rx = sense_cases(caseIdx, 1);
    Ry = sense_cases(caseIdx, 2);
    lambda = lambda_sense(caseIdx) * 1e-6;

    g = gfactor(map1, Rx, Ry, lambda);
    fieldName = sprintf('Rx%d_Ry%d', Rx, Ry);
    g_insufficient.(fieldName) = g;
    g_insufficient_stats.(fieldName) = gfactor_stats(g, map1);

    show_single_image(g, [0 5], ...
        sprintf('Part 2.8: g-factor Rx=%d, Ry=%d, lambda=%g', Rx, Ry, lambda), ...
        fullfile(figDir, sprintf('part2_8_gfactor_Rx%d_Ry%d.png', Rx, Ry)));

    fprintf('Part 2.8 g-factor Rx=%d Ry=%d lambda=%g: max = %.4f, mean = %.4f\n', ...
        Rx, Ry, lambda, g_insufficient_stats.(fieldName).max, g_insufficient_stats.(fieldName).mean);
end

%% Part 3.1 - calibration images
calib_cases = [8 8; 16 16; 32 32];
calib_data = struct();

for caseIdx = 1:size(calib_cases, 1)
    calibx = calib_cases(caseIdx, 1);
    caliby = calib_cases(caseIdx, 2);
    [imc, Mc] = imcalib(im, calibx, caliby);

    fieldName = sprintf('calib%d_%d', calibx, caliby);
    calib_data.(fieldName).imc = imc;
    calib_data.(fieldName).Mc = Mc;

    show_montage(abs(imc), [], sprintf('Part 3.1: calibration images %dx%d', calibx, caliby), ...
        fullfile(figDir, sprintf('part3_1_imc_%dx%d.png', calibx, caliby)));
    show_montage(kspace_spectrum(Mc), [], sprintf('Part 3.1: calibration k-space %dx%d', calibx, caliby), ...
        fullfile(figDir, sprintf('part3_1_Mc_%dx%d.png', calibx, caliby)));
end

%% Part 3.2 - undersampled images with calibration region
calibx = 32;
caliby = 32;
undersampled_calib_tests = struct();

for caseIdx = 1:size(sense_cases, 1)
    Rx = sense_cases(caseIdx, 1);
    Ry = sense_cases(caseIdx, 2);
    [imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);

    fieldName = sprintf('Rx%d_Ry%d', Rx, Ry);
    undersampled_calib_tests.(fieldName).imu = imu;
    undersampled_calib_tests.(fieldName).Mu = Mu;

    show_montage(abs(imu), [], sprintf('Part 3.2: undersampled+calib images Rx=%d, Ry=%d', Rx, Ry), ...
        fullfile(figDir, sprintf('part3_2_images_Rx%d_Ry%d.png', Rx, Ry)));
    show_montage(kspace_spectrum(Mu), [], sprintf('Part 3.2: undersampled+calib k-space Rx=%d, Ry=%d', Rx, Ry), ...
        fullfile(figDir, sprintf('part3_2_kspace_Rx%d_Ry%d.png', Rx, Ry)));
end

%% Part 3.3 - GRAPPA kernel weights
[~, Mc32] = imcalib(im, 32, 32);
kernel0 = grappa_calibrate(Mc32, 0);
fprintf('Part 3.3: GRAPPA kernel size = %d x %d\n', size(kernel0, 1), size(kernel0, 2));

show_matrix_image(abs(kernel0), [], 'Part 3.3: GRAPPA kernel magnitude, lambda=0', ...
    fullfile(figDir, 'part3_3_kernel_magnitude_lambda0.png'));
show_matrix_image(angle(kernel0), [-pi pi], 'Part 3.3: GRAPPA kernel phase, lambda=0', ...
    fullfile(figDir, 'part3_3_kernel_phase_lambda0.png'));

%% Part 3.4 - GRAPPA reconstruction for Rx=1, Ry=2, lambda=0
[~, Mc32] = imcalib(im, 32, 32);
[~, Mu_grappa] = undersamplecalib(im, 1, 2, 32, 32);
[imr0, Mr0] = grappa(Mu_grappa, Mc32, 0);

show_montage(abs(imr0), [], 'Part 3.4: GRAPPA coil images, lambda=0', ...
    fullfile(figDir, 'part3_4_grappa_coil_images_lambda0.png'));
show_montage(kspace_spectrum(Mr0), [], 'Part 3.4: GRAPPA reconstructed k-space, lambda=0', ...
    fullfile(figDir, 'part3_4_grappa_kspace_lambda0.png'));

grappa_sos0 = normalize95(sos_combine(imr0));
metrics_34 = compute_metrics(grappa_sos0, ref_grappa, max_val_grappa);

show_image_and_error(grappa_sos0, ref_grappa, max_val_grappa, ...
    'Part 3.4: GRAPPA SoS, lambda=0', ...
    fullfile(figDir, 'part3_4_grappa_sos_lambda0.png'));

print_metrics('Part 3.4 GRAPPA lambda=0', metrics_34);

%% Part 3.5 - GRAPPA reconstruction with selected regularization
lambda_candidates_grappa = [0, 10.^(-8:15)];
[~, Mc32] = imcalib(im, 32, 32);
[~, Mu_grappa] = undersamplecalib(im, 1, 2, 32, 32);

candidateImages = cell(numel(lambda_candidates_grappa), 1);
candidateKspace = cell(numel(lambda_candidates_grappa), 1);
grappa_scores = zeros(numel(lambda_candidates_grappa), 4);

for k = 1:numel(lambda_candidates_grappa)
    lambda = lambda_candidates_grappa(k);
    [imr, Mr] = grappa(Mu_grappa, Mc32, lambda);
    grappa_sos = normalize95(sos_combine(imr));
    metrics = compute_metrics(grappa_sos, ref_grappa, max_val_grappa);

    candidateImages{k} = grappa_sos;
    candidateKspace{k} = Mr;
    grappa_scores(k, 1:3) = [lambda, metrics.psnr, metrics.ssim];
end

psnrVals = grappa_scores(:, 2);
ssimVals = grappa_scores(:, 3);
if max(psnrVals) - min(psnrVals) < 1e-3
    psnrScore = zeros(size(psnrVals));
else
    psnrScore = (psnrVals - min(psnrVals)) ./ (max(psnrVals) - min(psnrVals));
end
if max(ssimVals) - min(ssimVals) < 1e-4
    ssimScore = zeros(size(ssimVals));
else
    ssimScore = (ssimVals - min(ssimVals)) ./ (max(ssimVals) - min(ssimVals));
end
grappa_scores(:, 4) = 0.5 * psnrScore + 0.5 * ssimScore;

[~, bestIdx] = max(grappa_scores(:, 4));
bestLambda = grappa_scores(bestIdx, 1);
bestImage = candidateImages{bestIdx};
bestKspace = candidateKspace{bestIdx};
bestMetrics = struct('psnr', grappa_scores(bestIdx, 2), 'ssim', grappa_scores(bestIdx, 3));
lambda_grappa_selected = bestLambda;
grappa_score_table = array2table(grappa_scores, ...
    'VariableNames', {'lambda', 'PSNR_dB', 'SSIM', 'combined_score'});

show_montage(kspace_spectrum(bestKspace), [], ...
    sprintf('Part 3.5: GRAPPA reconstructed k-space, lambda=%g', bestLambda), ...
    fullfile(figDir, 'part3_5_grappa_kspace_selected.png'));
show_image_and_error(bestImage, ref_grappa, max_val_grappa, ...
    sprintf('Part 3.5: GRAPPA SoS, lambda=%g', bestLambda), ...
    fullfile(figDir, 'part3_5_grappa_sos_selected.png'));

fprintf('\nPart 3.5 lambda search table for GRAPPA:\n');
disp(grappa_score_table);
fprintf('Part 3.5 selected lambda = %g\n', bestLambda);
print_metrics('Part 3.5 GRAPPA selected lambda', bestMetrics);

%% Local helper functions
function y = fft2c(x)
    y = fftshift(fftshift(fft2(ifftshift(ifftshift(x, 1), 2)), 1), 2);
end

function y = ifft2c(x)
    y = fftshift(fftshift(ifft2(ifftshift(ifftshift(x, 1), 2)), 1), 2);
end

function s = kspace_spectrum(M)
    s = log(1 + abs(M));
end

function out = normalize95(x)
    out = abs(x);
    out(~isfinite(out)) = 0;
    p = prctile(out(:), 95);
    if p > 0
        out = out ./ p;
    end
    out(~isfinite(out)) = 0;
end

function out = olc_recon(im, map)
    numerator = sum(conj(map) .* im, 3);
    denominator = sum(abs(map).^2, 3);
    out = numerator ./ denominator;
    out(~isfinite(out)) = 0;
end

function out = sos_combine(im)
    out = sqrt(sum(abs(im).^2, 3));
    out(~isfinite(out)) = 0;
end

function [imu, Mu] = undersample(im, Rx, Ry)
    [Nx, Ny, ~] = size(im);
    M = fft2c(im);
    xidx = centered_sampling_indices(Nx, Rx);
    yidx = centered_sampling_indices(Ny, Ry);
    Mu = M(xidx, yidx, :);
    imu = ifft2c(Mu);
    imu(~isfinite(imu)) = 0;
end

function [imc, Mc] = imcalib(im, calibx, caliby)
    [Nx, Ny, Nc] = size(im);
    M = fft2c(im);
    mask = false(Nx, Ny);
    xidx = center_indices(Nx, calibx);
    yidx = center_indices(Ny, caliby);
    mask(xidx, yidx) = true;
    Mc = M .* repmat(mask, [1 1 Nc]);
    imc = ifft2c(Mc);
    imc(~isfinite(imc)) = 0;
end

function [imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby)
    [Nx, Ny, Nc] = size(im);
    M = fft2c(im);
    mask = false(Nx, Ny);
    xSample = centered_sampling_indices(Nx, Rx);
    ySample = centered_sampling_indices(Ny, Ry);
    mask(xSample, ySample) = true;
    xidx = center_indices(Nx, calibx);
    yidx = center_indices(Ny, caliby);
    mask(xidx, yidx) = true;
    Mu = M .* repmat(mask, [1 1 Nc]);
    imu = ifft2c(Mu);
    imu(~isfinite(imu)) = 0;
end

function im_sense = l2sense(imu, map, Rx, Ry, lambda)
    [Nx, Ny, Nc] = size(map);
    Nxu = size(imu, 1);
    Nyu = size(imu, 2);
    if Nxu ~= Nx / Rx || Nyu ~= Ny / Ry
        error('The aliased image size is inconsistent with Rx and Ry.');
    end
    im_sense_complex = zeros(Nx, Ny);

    for ax = 1:Nxu
        srcx = alias_source_indices(ax, Nx, Rx);
        for ay = 1:Nyu
            srcy = alias_source_indices(ay, Ny, Ry);
            [XX, YY] = ndgrid(srcx, srcy);
            lin = sub2ind([Nx Ny], XX(:), YY(:));
            nsrc = numel(lin);
            if nsrc ~= Rx * Ry
                error('Unexpected number of unfolded source voxels.');
            end

            C = zeros(Nc, nsrc);
            for coil = 1:Nc
                tmp = map(:, :, coil);
                C(coil, :) = tmp(lin).';
            end

            y = squeeze(imu(ax, ay, :));
            y(~isfinite(y)) = 0;
            if norm(C, 'fro') == 0 || ~any(y)
                im_sense_complex(lin) = 0;
                continue;
            end

            G = C' * C;
            rhs = C' * y;
            if lambda == 0
                xhat = pinv(G) * rhs;
            else
                xhat = (G + lambda * eye(nsrc)) \ rhs;
            end
            im_sense_complex(lin) = xhat;
        end
    end

    im_sense = normalize95(abs(im_sense_complex));
end

function g = gfactor(map, Rx, Ry, lambda)
    [Nx, Ny, Nc] = size(map);
    Nxu = Nx / Rx;
    Nyu = Ny / Ry;
    g = zeros(Nx, Ny);

    for ax = 1:Nxu
        srcx = alias_source_indices(ax, Nx, Rx);
        for ay = 1:Nyu
            srcy = alias_source_indices(ay, Ny, Ry);
            [XX, YY] = ndgrid(srcx, srcy);
            lin = sub2ind([Nx Ny], XX(:), YY(:));
            nsrc = numel(lin);
            if nsrc ~= Rx * Ry
                error('Unexpected number of unfolded source voxels.');
            end

            C = zeros(Nc, nsrc);
            for coil = 1:Nc
                tmp = map(:, :, coil);
                C(coil, :) = tmp(lin).';
            end

            G = C' * C;
            B = pinv(G + lambda * eye(nsrc));
            covSense = B * G * B';

            for s = 1:nsrc
                sensPower = real(G(s, s));
                if sensPower <= 0
                    g(lin(s)) = 0;
                else
                    gArg = real(sensPower * covSense(s, s));
                    if gArg < 0 && abs(gArg) < 1e-10
                        gArg = 0;
                    end
                    gval = sqrt(max(gArg, 0));
                    if isfinite(gval)
                        g(lin(s)) = gval;
                    end
                end
            end
        end
    end

    allZero = sum(abs(map).^2, 3) == 0;
    g(allZero) = 0;
end

function stats = gfactor_stats(g, map)
    objectMask = sum(abs(map).^2, 3) > 0;
    vals = g(objectMask);
    vals = vals(isfinite(vals));
    if isempty(vals)
        stats.max = 0;
        stats.mean = 0;
    else
        stats.max = max(vals(:));
        stats.mean = mean(vals(:));
    end
end

function kernel = grappa_calibrate(Mc, lambda)
    [Nx, Ny, Nc] = size(Mc);
    mask = any(abs(Mc) > 0, 3);
    rows = find(any(mask, 2));
    cols = find(any(mask, 1));

    if isempty(rows) || isempty(cols)
        error('Calibration data are empty.');
    end

    xRange = rows(1):rows(end);
    yRange = cols(1):cols(end);
    xTargets = xRange(2:end-1);
    yTargets = yRange(2:end-1);
    acquiredY = centered_sampling_indices(Ny, 2);
    yTargets = yTargets(~ismember(yTargets, acquiredY));

    nExamples = numel(xTargets) * numel(yTargets);
    source = zeros(nExamples, 6 * Nc);
    target = zeros(nExamples, Nc);

    row = 0;
    for x = xTargets
        for y = yTargets
            row = row + 1;
            vec = zeros(1, 6 * Nc);
            for coil = 1:Nc
                base = (coil - 1) * 6;
                vec(base + (1:6)) = grappa_source_samples(Mc, x, y, coil);
            end
            source(row, :) = vec;
            target(row, :) = reshape(Mc(x, y, :), 1, Nc);
        end
    end

    kernel = (source' * source + lambda * eye(6 * Nc)) \ (source' * target);
end

function [imr, Mr] = grappa(Mu, Mc, lambda)
    [Nx, Ny, Nc] = size(Mu);
    kernel = grappa_calibrate(Mc, lambda);
    Mr = Mu;

    calibMask = any(abs(Mc) > 0, 3);
    rows = find(any(calibMask, 2));
    cols = find(any(calibMask, 1));

    acquiredMask = false(Nx, Ny);
    acquiredMask(:, centered_sampling_indices(Ny, 2)) = true;
    acquiredMask(rows(1):rows(end), cols(1):cols(end)) = true;

    for x = 2:Nx-1
        for y = 2:Ny-1
            if ~acquiredMask(x, y)
                vec = zeros(1, 6 * Nc);
                for coil = 1:Nc
                    base = (coil - 1) * 6;
                    vec(base + (1:6)) = grappa_source_samples(Mr, x, y, coil);
                end
                Mr(x, y, :) = reshape(vec * kernel, 1, 1, Nc);
            end
        end
    end

    imr = ifft2c(Mr);
    imr(~isfinite(imr)) = 0;
end

function samples = grappa_source_samples(M, x, y, coil)
    samples = [ ...
        M(x-1, y-1, coil), M(x, y-1, coil), M(x+1, y-1, coil), ...
        M(x-1, y+1, coil), M(x, y+1, coil), M(x+1, y+1, coil)];
end

function idx = center_indices(N, width)
    first = floor(N / 2) - floor(width / 2) + 1;
    idx = first:(first + width - 1);
end

function idx = centered_sampling_indices(N, R)
    if mod(N, R) ~= 0
        error('Image size must be divisible by the acceleration factor.');
    end
    center = floor(N / 2) + 1;
    idx = find(mod((1:N) - center, R) == 0);
    if numel(idx) ~= N / R
        error('Unexpected number of retained k-space samples.');
    end
end

function idx = alias_source_indices(aliasIndex, N, R)
    M = N / R;
    fullCenter = floor(N / 2) + 1;
    aliasCenter = floor(M / 2) + 1;
    first = mod(aliasIndex - aliasCenter + fullCenter - 1, M) + 1;
    idx = first:M:N;
end

function metrics = compute_metrics(img, ref, maxVal)
    a = real(img) ./ maxVal;
    b = real(ref) ./ maxVal;
    a(~isfinite(a)) = 0;
    b(~isfinite(b)) = 0;

    err = a - b;
    mse = mean(err(:).^2);
    if mse == 0
        metrics.psnr = Inf;
    else
        metrics.psnr = 10 * log10(1 / mse);
    end

    try
        metrics.ssim = ssim(a, b, 'DynamicRange', 1);
    catch
        try
            metrics.ssim = ssim(a, b);
        catch
            metrics.ssim = NaN;
            warning('SSIM could not be computed. Check that the Image Processing Toolbox is available.');
        end
    end
end

function print_metrics(label, metrics)
    fprintf('%s: PSNR = %.4f dB, SSIM = %.6f\n', label, metrics.psnr, metrics.ssim);
end

function show_montage(stack, displayRange, figTitle, outPath)
    figure('Color', 'w');
    if isempty(displayRange)
        hi = prctile(abs(stack(:)), 99);
        if hi <= 0 || ~isfinite(hi)
            hi = 1;
        end
        displayRange = [0 hi];
    end
    montage(stack, 'DisplayRange', displayRange);
    colormap gray;
    title(figTitle, 'Interpreter', 'none');
    save_current_figure(outPath);
end

function show_single_image(img, displayRange, figTitle, outPath)
    figure('Color', 'w');
    imshow(img, displayRange);
    colormap gray;
    colorbar;
    title(figTitle, 'Interpreter', 'none');
    save_current_figure(outPath);
end

function show_image_and_error(img, ref, maxVal, figTitle, outPath)
    figure('Color', 'w');
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    imshow(img, [0 maxVal]);
    colormap gray;
    title('Reconstruction', 'Interpreter', 'none');
    colorbar;

    nexttile;
    imshow(abs(img - ref), [0 0.2 * maxVal]);
    colormap gray;
    title('Absolute error', 'Interpreter', 'none');
    colorbar;

    sgtitle(figTitle, 'Interpreter', 'none');
    save_current_figure(outPath);
end

function show_matrix_image(A, displayRange, figTitle, outPath)
    figure('Color', 'w');
    if isempty(displayRange)
        imagesc(A);
    else
        imagesc(A, displayRange);
    end
    axis image;
    colormap gray;
    colorbar;
    title(figTitle, 'Interpreter', 'none');
    xlabel('Output coil');
    ylabel('Kernel coefficient index');
    save_current_figure(outPath);
end

function save_current_figure(outPath)
    [outDir, ~, ~] = fileparts(outPath);
    ensure_dir(outDir);
    try
        exportgraphics(gcf, outPath, 'Resolution', 200);
    catch
        saveas(gcf, outPath);
    end
end

function ensure_dir(pathName)
    if ~exist(pathName, 'dir')
        mkdir(pathName);
    end
end

