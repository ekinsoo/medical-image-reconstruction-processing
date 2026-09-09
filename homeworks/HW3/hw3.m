%% 1.1 – Kalibrasyon verisini gorsellestirme

load('SM_25nm_50x50.mat');

locs = [5, 500, 1500];

fig = figure('Name','1.1','NumberTitle','off','Position',[50 50 900 600]);
for i = 1:3
    subplot(3,1,i);
    plot(abs(SM(1,:,locs(i))), 'LineWidth', 1);
    xlabel('Frequency Index'); ylabel('|SM|');
    title(['Coil 1 – Grid Location ' num2str(locs(i))]);
    grid on;
end
sgtitle('Part 1.1: Magnitude Spectra – Coil 1', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_1_1_coil1.png'));

fig = figure('Name','1.1b','NumberTitle','off','Position',[50 50 900 600]);
for i = 1:3
    subplot(3,1,i);
    plot(abs(SM(2,:,locs(i))), 'LineWidth', 1);
    xlabel('Frequency Index'); ylabel('|SM|');
    title(['Coil 2 – Grid Location ' num2str(locs(i))]);
    grid on;
end
sgtitle('Part 1.1: Magnitude Spectra – Coil 2', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_1_1_coil2.png'));

fprintf('Part 1.1 done – figures saved.\n');

%% 1.2 – Sistem matrisi S'yi hazirlama

S = SM(:, 1:end/2, :);         % spektrumun ikinci yarisini at
S = reshape(S, 2*816, 2500);   % iki coili birlestir
S = [real(S); imag(S)];        % reel ve imajiner parcalari birlestir -> 3264x2500

fprintf('Size of S: %d x %d\n', size(S,1), size(S,2));

fig = figure('Name','1.2','NumberTitle','off','Position',[50 50 900 600]);
for i = 1:3
    subplot(3,1,i);
    plot(S(:, locs(i)), 'LineWidth', 1);
    xlabel('Row Index'); ylabel('Value');
    title(['Column ' num2str(locs(i)) ' of S']);
    grid on;
end
sgtitle('Part 1.2: Columns of System Matrix S', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_1_2_S_columns.png'));

save('S.mat', 'S');
info_S = dir('S.mat');
fprintf('Size of S.mat: %.2f MB\n', info_S.bytes / 1e6);

%% 1.3 – Olcum vektoru u'yu hazirlama

load('meas_25nm_phantom.mat');

fig = figure('Name','1.3','NumberTitle','off','Position',[50 50 500 450]);
imagesc(phantom_ref); colormap(gray); axis image; colorbar;
title('Reference Phantom m_{ref}', 'FontWeight','bold');
xlabel('x'); ylabel('y');
saveas(fig, fullfile(figDir, 'fig_1_3_phantom.png'));

max_val = max(phantom_ref(:));
fprintf('max_val = %.4f\n', max_val);

% u icin de S ile ayni adimlar
u = meas(:, 1:end/2);
u = reshape(u, 2*816, 1);
u = [real(u); imag(u)];       % 3264x1

fprintf('Size of u: %d x %d\n', size(u,1), size(u,2));

fig = figure('Name','1.3b','NumberTitle','off','Position',[50 50 900 350]);
plot(u, 'LineWidth', 1);
xlabel('Index'); ylabel('u');
title('Part 1.3: Measurement Vector u', 'FontWeight','bold');
grid on;
saveas(fig, fullfile(figDir, 'fig_1_3_u.png'));

save('u.mat', 'u');
info_u = dir('u.mat');
fprintf('Size of u.mat: %.2f KB\n', info_u.bytes / 1e3);

ref_norm = double(phantom_ref) / max_val;

%% 2.1 – SVD

fprintf('\nComputing compact SVD of S...\n');
[U, Sigma, V] = svd(S, 'econ');
sigma_vals = diag(Sigma);

fig = figure('Name','2.1','NumberTitle','off','Position',[50 50 800 400]);
semilogy(sigma_vals, 'LineWidth', 1.5);
xlabel('Index'); ylabel('Singular value');
title('Part 2.1 - Singular values of S', 'FontWeight','bold');
grid on;
saveas(fig, fullfile(figDir, 'fig_2_1_singular_values.png'));

% Yakinlastirilmis: ilk 200 SV
fig = figure('Name','2.1z','NumberTitle','off','Position',[50 50 800 400]);
semilogy(1:200, sigma_vals(1:200), 'LineWidth', 1.5);
xlabel('Index'); ylabel('Singular value');
title('Part 2.1 - Singular values of S (zoomed, first 200)', 'FontWeight','bold');
grid on;
saveas(fig, fullfile(figDir, 'fig_2_1_singular_values_zoomed.png'));

cond_svd = sigma_vals(1) / sigma_vals(end);
fprintf('Condition number (SVD):   %.4e\n', cond_svd);
fprintf('Condition number (cond): %.4e\n', cond(S));

%% 2.2 – Full SVD (Moore-Penrose pseudo-inverse)

c_full = V * (diag(1./sigma_vals) * (U' * u));
ima_full = reshape(c_full, 50, 50);
ima_full(ima_full < 0) = 0;

ima_full_norm = double(ima_full) / max_val;
psnr_22 = psnr(ima_full_norm, ref_norm);
ssim_22 = ssim(ima_full_norm, ref_norm);
fprintf('\nPart 2.2 – Full SVD: PSNR = %.2f dB, SSIM = %.4f\n', psnr_22, ssim_22);

error_22 = abs(ima_full - phantom_ref);

fig = figure('Name','2.2','NumberTitle','off','Position',[50 50 1100 380]);
colormap(gray);
subplot(1,3,1); imshow(ima_full,  [0 max_val]);      colorbar;
                title('Full SVD Image');              axis on;
subplot(1,3,2); imshow(phantom_ref, [0 max_val]);    colorbar;
                title('Reference m_{ref}');           axis on;
subplot(1,3,3); imshow(error_22, [0 0.5*max_val]);  colorbar;
                title('Error Image');                 axis on;
sgtitle(sprintf('Part 2.2: Full SVD  |  PSNR = %.1f dB, SSIM = %.4f', ...
        psnr_22, ssim_22), 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_2_2_full_svd.png'));

%% 2.3 – Truncated SVD (cond ~ 100)

target_cond_23 = 100;
M23 = sum(sigma_vals / sigma_vals(1) >= 1/target_cond_23);
fprintf('\nPart 2.3: M = %d singular values kept ', M23);
fprintf('(actual cond = %.2f)\n', sigma_vals(1)/sigma_vals(M23));

c_23 = V(:,1:M23) * (diag(1./sigma_vals(1:M23)) * (U(:,1:M23)' * u));
ima_23 = reshape(c_23, 50, 50);
ima_23(ima_23 < 0) = 0;

ima_23_norm = double(ima_23) / max_val;
psnr_23 = psnr(ima_23_norm, ref_norm);
ssim_23 = ssim(ima_23_norm, ref_norm);
fprintf('Part 2.3 – TSVD (cond~100): PSNR = %.2f dB, SSIM = %.4f\n', psnr_23, ssim_23);

error_23 = abs(ima_23 - phantom_ref);

fig = figure('Name','2.3','NumberTitle','off','Position',[50 50 1100 380]);
colormap(gray);
subplot(1,3,1); imshow(ima_23,    [0 max_val]);      colorbar;
                title(sprintf('TSVD (cond~100, M=%d)',M23)); axis on;
subplot(1,3,2); imshow(phantom_ref, [0 max_val]);    colorbar;
                title('Reference m_{ref}');           axis on;
subplot(1,3,3); imshow(error_23, [0 0.5*max_val]);  colorbar;
                title('Error Image');                 axis on;
sgtitle(sprintf('Part 2.3: TSVD (cond~100)  |  PSNR = %.1f dB, SSIM = %.4f', ...
        psnr_23, ssim_23), 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_2_3_tsvd.png'));

%% 2.4 – Farkli condition number'lar icin TSVD

cond_list = [1, 5, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7];
N_cond    = length(cond_list);
psnr_cond = zeros(1, N_cond);
ssim_cond = zeros(1, N_cond);
M_cond    = zeros(1, N_cond);

for k = 1:N_cond
    tgt  = cond_list(k);
    Mk   = sum(sigma_vals / sigma_vals(1) >= 1/tgt);
    if Mk < 1; Mk = 1; end
    M_cond(k) = Mk;

    c_k   = V(:,1:Mk) * (diag(1./sigma_vals(1:Mk)) * (U(:,1:Mk)' * u));
    ima_k = reshape(c_k, 50, 50);
    ima_k(ima_k < 0) = 0;

    ima_k_norm    = double(ima_k) / max_val;
    psnr_cond(k)  = psnr(ima_k_norm, ref_norm);
    ssim_cond(k)  = ssim(ima_k_norm, ref_norm);

    err_k = abs(ima_k - phantom_ref);

    fig = figure('Name',sprintf('2.4: cond=%g',tgt),'NumberTitle','off','Position',[50 50 1100 380]);
    colormap(gray);
    subplot(1,3,1); imshow(ima_k, [0 max_val]);         colorbar;
                    title(sprintf('TSVD cond=%g, M=%d', tgt, Mk)); axis on;
    subplot(1,3,2); imshow(phantom_ref, [0 max_val]);   colorbar;
                    title('Reference m_{ref}');          axis on;
    subplot(1,3,3); imshow(err_k, [0 0.5*max_val]);    colorbar;
                    title('Error Image');                axis on;
    sgtitle(sprintf('Part 2.4: cond=%g  |  PSNR=%.1f dB, SSIM=%.4f', ...
            tgt, psnr_cond(k), ssim_cond(k)), 'FontWeight','bold');
    saveas(fig, fullfile(figDir, sprintf('fig_2_4_cond_%g.png', tgt)));
    close(fig);
end

fprintf('\nPart 2.4 Summary:\n');
fprintf('%-14s %-8s %-12s %-8s\n', 'Cond. Number','M','PSNR (dB)','SSIM');
fprintf('%s\n', repmat('-',1,46));
for k = 1:N_cond
    fprintf('%-14g %-8d %-12.2f %-8.4f\n', ...
        cond_list(k), M_cond(k), psnr_cond(k), ssim_cond(k));
end

fig = figure('Name','2.4 metrics','NumberTitle','off','Position',[50 50 700 520]);
subplot(2,1,1);
semilogx(cond_list, psnr_cond, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
xlabel('Condition Number'); ylabel('PSNR (dB)');
title('PSNR vs Condition Number'); grid on;
subplot(2,1,2);
semilogx(cond_list, ssim_cond, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
xlabel('Condition Number'); ylabel('SSIM');
title('SSIM vs Condition Number'); grid on;
sgtitle('Part 2.4: Image Quality vs Condition Number', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_2_4_psnr_ssim.png'));

[~, best_idx_24] = max(psnr_cond);
fprintf('\nBest condition number (by PSNR): %g  ', cond_list(best_idx_24));
fprintf('(PSNR = %.2f dB, SSIM = %.4f)\n', psnr_cond(best_idx_24), ssim_cond(best_idx_24));

%% 2.6 – Filtered SVD (lambda = sigma_1 * sigma_N)

lambda_26 = sigma_vals(1) * sigma_vals(end);
fprintf('\nPart 2.6: lambda = sigma_1 * sigma_N = %.4e\n', lambda_26);

% Filtrelenmis tekil degerler
sigma_filt_26 = (sigma_vals.^2 + lambda_26) ./ sigma_vals;

fig = figure('Name','2.6','NumberTitle','off','Position',[50 50 800 400]);
semilogy(sigma_vals, 'b-', 'LineWidth', 1.5); hold on;
semilogy(sigma_filt_26, 'r-', 'LineWidth', 1.5);
legend('Original','Filtered','Location','best');
xlabel('Index'); ylabel('Singular value');
title('Part 2.6 - Original and filtered singular values', 'FontWeight','bold');
grid on;
saveas(fig, fullfile(figDir, 'fig_2_6_filtered_sv.png'));

% Yakinlastirilmis: son 800 SV (gecis bolgesi)
zoom_start = max(1, length(sigma_vals) - 799);
fig = figure('Name','2.6z','NumberTitle','off','Position',[50 50 800 400]);
semilogy(zoom_start:length(sigma_vals), sigma_vals(zoom_start:end), 'b-', 'LineWidth', 1.5); hold on;
semilogy(zoom_start:length(sigma_vals), sigma_filt_26(zoom_start:end), 'r-', 'LineWidth', 1.5);
legend('Original','Filtered','Location','best');
xlabel('Index'); ylabel('Singular value');
title('Part 2.6 - Original vs Filtered SV (zoomed, transition region)', 'FontWeight','bold');
grid on;
saveas(fig, fullfile(figDir, 'fig_2_6_filtered_sv_zoomed.png'));

% Goruntu olusturma
c_26   = V * (diag(1./sigma_filt_26) * (U' * u));
ima_26 = reshape(c_26, 50, 50);
ima_26(ima_26 < 0) = 0;

ima_26_norm = double(ima_26) / max_val;
psnr_26 = psnr(ima_26_norm, ref_norm);
ssim_26 = ssim(ima_26_norm, ref_norm);
fprintf('Part 2.6 – Filtered SVD: PSNR = %.2f dB, SSIM = %.4f\n', psnr_26, ssim_26);

error_26 = abs(ima_26 - phantom_ref);

fig = figure('Name','2.6 img','NumberTitle','off','Position',[50 50 1100 380]);
colormap(gray);
subplot(1,3,1); imshow(ima_26,    [0 max_val]);      colorbar;
                title(sprintf('Filtered SVD (\\lambda=%.1e)', lambda_26)); axis on;
subplot(1,3,2); imshow(phantom_ref, [0 max_val]);    colorbar;
                title('Reference m_{ref}');           axis on;
subplot(1,3,3); imshow(error_26, [0 0.5*max_val]);  colorbar;
                title('Error Image');                 axis on;
sgtitle(sprintf('Part 2.6: Filtered SVD  |  PSNR = %.1f dB, SSIM = %.4f', ...
        psnr_26, ssim_26), 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_2_6_image.png'));

%% 2.7 – Optimal lambda (PSNR/SSIM ile arama)

exp_low  = floor(log10(sigma_vals(end)^2));
exp_high = ceil(log10(sigma_vals(1)^2));
lambda_exp_arr = exp_low : exp_high;
lambda_arr     = 10.^lambda_exp_arr;
N_lam          = length(lambda_arr);

psnr_lam = zeros(1, N_lam);
ssim_lam = zeros(1, N_lam);

for k = 1:N_lam
    lam   = lambda_arr(k);
    sf    = (sigma_vals.^2 + lam) ./ sigma_vals;
    c_lam = V * (diag(1./sf) * (U' * u));
    ima_lam = reshape(c_lam, 50, 50);
    ima_lam(ima_lam < 0) = 0;
    ima_lam_norm = double(ima_lam) / max_val;
    psnr_lam(k)  = psnr(ima_lam_norm, ref_norm);
    ssim_lam(k)  = ssim(ima_lam_norm, ref_norm);
end

fig = figure('Name','2.7','NumberTitle','off','Position',[50 50 700 520]);
subplot(2,1,1);
semilogx(lambda_arr, psnr_lam, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
xlabel('\lambda'); ylabel('PSNR (dB)');
title('PSNR vs \lambda'); grid on;
subplot(2,1,2);
semilogx(lambda_arr, ssim_lam, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
xlabel('\lambda'); ylabel('SSIM');
title('SSIM vs \lambda'); grid on;
sgtitle('Part 2.7: Image Quality vs Regularization \lambda', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_2_7_psnr_ssim_lambda.png'));

[~, best_lam_idx] = max(psnr_lam);
lambda_star = lambda_arr(best_lam_idx);
fprintf('\nPart 2.7: Optimal lambda* = %.2e (10^%d)\n', ...
        lambda_star, lambda_exp_arr(best_lam_idx));

% lambda*/10, lambda*, 10*lambda* icin goruntu karsilastirmasi
lam_cases   = [lambda_star/10, lambda_star, 10*lambda_star];
case_strs   = {'\lambda^*/10', '\lambda^*', '10\lambda^*'};
fig_names27 = {'fig_2_7_lam_div10.png', 'fig_2_7_lam_star.png', 'fig_2_7_lam_x10.png'};

for k = 1:3
    lam   = lam_cases(k);
    sf    = (sigma_vals.^2 + lam) ./ sigma_vals;
    c_lam = V * (diag(1./sf) * (U' * u));
    ima_lam = reshape(c_lam, 50, 50);
    ima_lam(ima_lam < 0) = 0;
    ima_lam_norm = double(ima_lam) / max_val;
    p = psnr(ima_lam_norm, ref_norm);
    s = ssim(ima_lam_norm, ref_norm);

    fig = figure('Name',sprintf('2.7: %s',case_strs{k}),'NumberTitle','off','Position',[50 50 1100 380]);
    colormap(gray);
    subplot(1,3,1); imshow(ima_lam, [0 max_val]);               colorbar;
                    title(sprintf('Filtered SVD (%s)', case_strs{k})); axis on;
    subplot(1,3,2); imshow(phantom_ref, [0 max_val]);            colorbar;
                    title('Reference m_{ref}');                   axis on;
    subplot(1,3,3); imshow(abs(ima_lam-phantom_ref), [0 0.5*max_val]); colorbar;
                    title('Error Image');                         axis on;
    sgtitle(sprintf('Part 2.7: %s (\\lambda=%.1e)  |  PSNR=%.1f dB, SSIM=%.4f', ...
            case_strs{k}, lam, p, s), 'FontWeight','bold');
    saveas(fig, fullfile(figDir, fig_names27{k}));
    fprintf('%s (lambda=%.2e): PSNR=%.2f dB, SSIM=%.4f\n', case_strs{k}, lam, p, s);
end

%% 2.8 – L-curve

res_norms = zeros(1, N_lam);
sol_norms = zeros(1, N_lam);

for k = 1:N_lam
    lam   = lambda_arr(k);
    sf    = (sigma_vals.^2 + lam) ./ sigma_vals;
    c_lam = V * (diag(1./sf) * (U' * u));
    res_norms(k) = norm(S * c_lam - u);
    sol_norms(k) = norm(c_lam);
end

fig = figure('Name','2.8','NumberTitle','off','Position',[50 50 620 520]);
loglog(res_norms, sol_norms, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
hold on;
for k = 1:N_lam
    text(res_norms(k)*1.05, sol_norms(k), ...
         sprintf('10^{%d}', lambda_exp_arr(k)), 'FontSize', 7);
end
xlabel('||Sc - u||_2  (Residual Norm)');
ylabel('||c||_2  (Solution Norm)');
title('Part 2.8: L-curve', 'FontWeight','bold');
grid on;
saveas(fig, fullfile(figDir, 'fig_2_8_lcurve.png'));

% Menger egrilik formuluyle kose noktasini bul
log_res = log(res_norms);
log_sol = log(sol_norms);
curvature = zeros(1, N_lam);
for k = 2:N_lam-1
    x1 = log_res(k-1); y1 = log_sol(k-1);
    x2 = log_res(k);   y2 = log_sol(k);
    x3 = log_res(k+1); y3 = log_sol(k+1);
    twice_area = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1));
    d12 = sqrt((x2-x1)^2 + (y2-y1)^2);
    d23 = sqrt((x3-x2)^2 + (y3-y2)^2);
    d13 = sqrt((x3-x1)^2 + (y3-y1)^2);
    if d12*d23*d13 > 0
        curvature(k) = twice_area / (d12 * d23 * d13);
    end
end
[~, corner_idx] = max(curvature);
lambda_lcurve = lambda_arr(corner_idx);
fprintf('\nPart 2.8: L-curve corner at lambda = %.2e (10^%d)\n', ...
        lambda_lcurve, lambda_exp_arr(corner_idx));

% L-curve lambda* ile goruntu olustur
sf_lc = (sigma_vals.^2 + lambda_lcurve) ./ sigma_vals;
c_lc  = V * (diag(1./sf_lc) * (U' * u));
ima_lc = reshape(c_lc, 50, 50);
ima_lc(ima_lc < 0) = 0;
ima_lc_norm = double(ima_lc) / max_val;
psnr_lc = psnr(ima_lc_norm, ref_norm);
ssim_lc = ssim(ima_lc_norm, ref_norm);
fprintf('L-curve lambda*: PSNR = %.2f dB, SSIM = %.4f\n', psnr_lc, ssim_lc);

error_lc = abs(ima_lc - phantom_ref);
fig = figure('Name','2.8 img','NumberTitle','off','Position',[50 50 1100 380]);
colormap(gray);
subplot(1,3,1); imshow(ima_lc, [0 max_val]);      colorbar;
                title(sprintf('Filtered SVD (L-curve \\lambda=%.1e)', lambda_lcurve)); axis on;
subplot(1,3,2); imshow(phantom_ref, [0 max_val]);  colorbar;
                title('Reference m_{ref}');          axis on;
subplot(1,3,3); imshow(error_lc, [0 0.5*max_val]); colorbar;
                title('Error Image');                axis on;
sgtitle(sprintf('Part 2.8: L-curve \\lambda*  |  PSNR=%.1f dB, SSIM=%.4f', ...
        psnr_lc, ssim_lc), 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_2_8_lcurve_image.png'));

fprintf('\n=== Parts I & II complete. ===\n');

%% 3.1 – Satir normu esikleme

row_norms = sqrt(sum(S.^2, 2));

fig = figure('Name','3.1','NumberTitle','off','Position',[50 50 800 400]);
plot(row_norms, 'LineWidth', 1);
xlabel('Row Number'); ylabel('Row Norm');
title('Part 3.1: Row Norms of S', 'FontWeight','bold');
grid on;
saveas(fig, fullfile(figDir, 'fig_3_1_row_norms.png'));

max_norm = max(row_norms);
fprintf('\nPart 3.1: max_norm = %.4e\n', max_norm);

% max_norm/10 alti satirlari at
thresh_31 = max_norm / 10;
keep_idx = row_norms >= thresh_31;
S_thresh = S(keep_idx, :);
u_thresh = u(keep_idx);

fprintf('After thresholding (max_norm/10):\n');
fprintf('  Size of S: %d x %d\n', size(S_thresh,1), size(S_thresh,2));
fprintf('  Size of u: %d x %d\n', size(u_thresh,1), size(u_thresh,2));

%% 3.2 – Standard Kaczmarz (esiklenmis S ile)

fprintf('\nPart 3.2: Running Standard Kaczmarz (10 iterations)...\n');

N_iter = 10;
[n_rows, n_cols] = size(S_thresh);
c_kacz = zeros(n_cols, 1);

psnr_kacz = zeros(1, N_iter);
ssim_kacz = zeros(1, N_iter);

for iter = 1:N_iter
    order = randperm(n_rows);
    for sub = 1:n_rows
        idx = order(sub);
        row_i = S_thresh(idx, :);
        c_kacz = c_kacz + (u_thresh(idx) - row_i * c_kacz) / (row_i * row_i') * row_i';
    end
    c_kacz(c_kacz < 0) = 0;

    ima_kacz = reshape(c_kacz, 50, 50);
    ima_kacz_norm = double(ima_kacz) / max_val;
    psnr_kacz(iter) = psnr(ima_kacz_norm, ref_norm);
    ssim_kacz(iter) = ssim(ima_kacz_norm, ref_norm);

    err_kacz = abs(ima_kacz - phantom_ref);

    fig = figure('Name',sprintf('3.2: iter %d',iter),'NumberTitle','off','Position',[50 50 1100 380]);
    colormap(gray);
    subplot(1,3,1); imshow(ima_kacz, [0 max_val]);      colorbar;
                    title(sprintf('Kaczmarz iter %d', iter)); axis on;
    subplot(1,3,2); imshow(phantom_ref, [0 max_val]);    colorbar;
                    title('Reference m_{ref}');           axis on;
    subplot(1,3,3); imshow(err_kacz, [0 0.5*max_val]);  colorbar;
                    title('Error Image');                 axis on;
    sgtitle(sprintf('Part 3.2: Kaczmarz iter %d  |  PSNR=%.1f dB, SSIM=%.4f', ...
            iter, psnr_kacz(iter), ssim_kacz(iter)), 'FontWeight','bold');
    saveas(fig, fullfile(figDir, sprintf('fig_3_2_iter_%d.png', iter)));
    close(fig);

    fprintf('  Iter %d: PSNR = %.2f dB, SSIM = %.4f\n', iter, psnr_kacz(iter), ssim_kacz(iter));
end

fig = figure('Name','3.2 metrics','NumberTitle','off','Position',[50 50 700 520]);
subplot(2,1,1);
plot(1:N_iter, psnr_kacz, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
xlabel('Iteration'); ylabel('PSNR (dB)');
title('PSNR vs Iteration (Std Kaczmarz)'); grid on;
subplot(2,1,2);
plot(1:N_iter, ssim_kacz, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
xlabel('Iteration'); ylabel('SSIM');
title('SSIM vs Iteration (Std Kaczmarz)'); grid on;
sgtitle('Part 3.2: Standard Kaczmarz – Image Quality vs Iteration', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_3_2_psnr_ssim.png'));

%% 3.3 – Standard Kaczmarz, farkli esik degerleri

thresh_factors = [1e-4, 1e-3, 1e-2, 1e-1];
N_thresh = length(thresh_factors);
psnr_thresh = zeros(1, N_thresh);
ssim_thresh = zeros(1, N_thresh);

fprintf('\nPart 3.3: Standard Kaczmarz with different thresholds...\n');

for t = 1:N_thresh
    thresh_val = max_norm * thresh_factors(t);
    keep = row_norms >= thresh_val;
    S_t = S(keep, :);
    u_t = u(keep);

    c_t = zeros(n_cols, 1);

    for iter = 1:10
        order = randperm(size(S_t, 1));
        for sub = 1:size(S_t, 1)
            idx = order(sub);
            row_i = S_t(idx, :);
            c_t = c_t + (u_t(idx) - row_i * c_t) / (row_i * row_i') * row_i';
        end
        c_t(c_t < 0) = 0;
    end

    ima_t = reshape(c_t, 50, 50);
    ima_t_norm = double(ima_t) / max_val;
    psnr_thresh(t) = psnr(ima_t_norm, ref_norm);
    ssim_thresh(t) = ssim(ima_t_norm, ref_norm);

    err_t = abs(ima_t - phantom_ref);

    fig = figure('Name',sprintf('3.3: thresh=%g',thresh_factors(t)),'NumberTitle','off','Position',[50 50 1100 380]);
    colormap(gray);
    subplot(1,3,1); imshow(ima_t, [0 max_val]);         colorbar;
                    title(sprintf('Kaczmarz (thresh=%.0e)', thresh_factors(t))); axis on;
    subplot(1,3,2); imshow(phantom_ref, [0 max_val]);   colorbar;
                    title('Reference m_{ref}');           axis on;
    subplot(1,3,3); imshow(err_t, [0 0.5*max_val]);    colorbar;
                    title('Error Image');                 axis on;
    sgtitle(sprintf('Part 3.3: thresh=%.0e  |  PSNR=%.1f dB, SSIM=%.4f', ...
            thresh_factors(t), psnr_thresh(t), ssim_thresh(t)), 'FontWeight','bold');
    saveas(fig, fullfile(figDir, sprintf('fig_3_3_thresh_%g.png', thresh_factors(t))));
    close(fig);

    fprintf('  Threshold %.0e: PSNR = %.2f dB, SSIM = %.4f\n', ...
        thresh_factors(t), psnr_thresh(t), ssim_thresh(t));
end

fig = figure('Name','3.3 metrics','NumberTitle','off','Position',[50 50 700 520]);
subplot(2,1,1);
semilogx(thresh_factors, psnr_thresh, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
xlabel('Threshold (max\_norm orani)'); ylabel('PSNR (dB)');
title('PSNR vs Threshold'); grid on;
subplot(2,1,2);
semilogx(thresh_factors, ssim_thresh, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
xlabel('Threshold (max\_norm orani)'); ylabel('SSIM');
title('SSIM vs Threshold'); grid on;
sgtitle('Part 3.3: Image Quality vs Row-norm Threshold', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_3_3_psnr_ssim.png'));

%% 3.4 – Regularized Kaczmarz (esikleme yok, full S)

lambda_34 = sigma_vals(1) * sigma_vals(end);
fprintf('\nPart 3.4: Regularized Kaczmarz (lambda = sigma_1*sigma_N = %.4e)\n', lambda_34);

[n_rows_full, n_cols_full] = size(S);
c_rkacz = zeros(n_cols_full, 1);

psnr_rkacz = zeros(1, N_iter);
ssim_rkacz = zeros(1, N_iter);

for iter = 1:N_iter
    order = randperm(n_rows_full);
    for sub = 1:n_rows_full
        idx = order(sub);
        row_i = S(idx, :);
        c_rkacz = c_rkacz + (u(idx) - row_i * c_rkacz) / (row_i * row_i' + lambda_34) * row_i';
    end
    c_rkacz(c_rkacz < 0) = 0;

    ima_rkacz = reshape(c_rkacz, 50, 50);
    ima_rkacz_norm = double(ima_rkacz) / max_val;
    psnr_rkacz(iter) = psnr(ima_rkacz_norm, ref_norm);
    ssim_rkacz(iter) = ssim(ima_rkacz_norm, ref_norm);

    err_rkacz = abs(ima_rkacz - phantom_ref);

    fig = figure('Name',sprintf('3.4: iter %d',iter),'NumberTitle','off','Position',[50 50 1100 380]);
    colormap(gray);
    subplot(1,3,1); imshow(ima_rkacz, [0 max_val]);      colorbar;
                    title(sprintf('Reg. Kaczmarz iter %d', iter)); axis on;
    subplot(1,3,2); imshow(phantom_ref, [0 max_val]);    colorbar;
                    title('Reference m_{ref}');           axis on;
    subplot(1,3,3); imshow(err_rkacz, [0 0.5*max_val]);  colorbar;
                    title('Error Image');                  axis on;
    sgtitle(sprintf('Part 3.4: Reg. Kaczmarz iter %d  |  PSNR=%.1f dB, SSIM=%.4f', ...
            iter, psnr_rkacz(iter), ssim_rkacz(iter)), 'FontWeight','bold');
    saveas(fig, fullfile(figDir, sprintf('fig_3_4_iter_%d.png', iter)));
    close(fig);

    fprintf('  Iter %d: PSNR = %.2f dB, SSIM = %.4f\n', iter, psnr_rkacz(iter), ssim_rkacz(iter));
end

fig = figure('Name','3.4 metrics','NumberTitle','off','Position',[50 50 700 520]);
subplot(2,1,1);
plot(1:N_iter, psnr_rkacz, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
xlabel('Iteration'); ylabel('PSNR (dB)');
title('PSNR vs Iteration (Reg. Kaczmarz)'); grid on;
subplot(2,1,2);
plot(1:N_iter, ssim_rkacz, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
xlabel('Iteration'); ylabel('SSIM');
title('SSIM vs Iteration (Reg. Kaczmarz)'); grid on;
sgtitle('Part 3.4: Regularized Kaczmarz – Image Quality vs Iteration', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_3_4_psnr_ssim.png'));

%% 3.5 – Regularized Kaczmarz, farkli lambda_rel degerleri

lambda_rel_arr = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
N_lrel = length(lambda_rel_arr);
psnr_lrel = zeros(1, N_lrel);
ssim_lrel = zeros(1, N_lrel);

fprintf('\nPart 3.5: Regularized Kaczmarz with different lambda_rel...\n');

for t = 1:N_lrel
    lam = sigma_vals(1)^2 * lambda_rel_arr(t);
    c_t = zeros(n_cols_full, 1);

    for iter = 1:10
        order = randperm(n_rows_full);
        for sub = 1:n_rows_full
            idx = order(sub);
            row_i = S(idx, :);
            c_t = c_t + (u(idx) - row_i * c_t) / (row_i * row_i' + lam) * row_i';
        end
        c_t(c_t < 0) = 0;
    end

    ima_t = reshape(c_t, 50, 50);
    ima_t_norm = double(ima_t) / max_val;
    psnr_lrel(t) = psnr(ima_t_norm, ref_norm);
    ssim_lrel(t) = ssim(ima_t_norm, ref_norm);

    err_t = abs(ima_t - phantom_ref);

    fig = figure('Name',sprintf('3.5: lrel=%g',lambda_rel_arr(t)),'NumberTitle','off','Position',[50 50 1100 380]);
    colormap(gray);
    subplot(1,3,1); imshow(ima_t, [0 max_val]);         colorbar;
                    title(sprintf('Reg. Kaczmarz (\\lambda_{rel}=%.0e)', lambda_rel_arr(t))); axis on;
    subplot(1,3,2); imshow(phantom_ref, [0 max_val]);   colorbar;
                    title('Reference m_{ref}');           axis on;
    subplot(1,3,3); imshow(err_t, [0 0.5*max_val]);    colorbar;
                    title('Error Image');                 axis on;
    sgtitle(sprintf('Part 3.5: \\lambda_{rel}=%.0e  |  PSNR=%.1f dB, SSIM=%.4f', ...
            lambda_rel_arr(t), psnr_lrel(t), ssim_lrel(t)), 'FontWeight','bold');
    saveas(fig, fullfile(figDir, sprintf('fig_3_5_lrel_%g.png', lambda_rel_arr(t))));
    close(fig);

    fprintf('  lambda_rel = %.0e: PSNR = %.2f dB, SSIM = %.4f\n', ...
        lambda_rel_arr(t), psnr_lrel(t), ssim_lrel(t));
end

fig = figure('Name','3.5 metrics','NumberTitle','off','Position',[50 50 700 520]);
subplot(2,1,1);
semilogx(lambda_rel_arr, psnr_lrel, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
xlabel('\lambda_{rel}'); ylabel('PSNR (dB)');
title('PSNR vs \lambda_{rel}'); grid on;
subplot(2,1,2);
semilogx(lambda_rel_arr, ssim_lrel, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
xlabel('\lambda_{rel}'); ylabel('SSIM');
title('SSIM vs \lambda_{rel}'); grid on;
sgtitle('Part 3.5: Regularized Kaczmarz – Image Quality vs \lambda_{rel}', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_3_5_psnr_ssim.png'));

%% 3.6 – Gurultu ekleme + Standard Kaczmarz (Part 3.3 tekrari)

fprintf('\nPart 3.6: Adding noise and repeating Part 3.3...\n');

noise_std = max_norm / 1000;
u_noisy = u + noise_std * randn(size(u));

psnr_thresh_noisy = zeros(1, N_thresh);
ssim_thresh_noisy = zeros(1, N_thresh);

for t = 1:N_thresh
    thresh_val = max_norm * thresh_factors(t);
    keep = row_norms >= thresh_val;
    S_t = S(keep, :);
    u_t = u_noisy(keep);

    c_t = zeros(n_cols, 1);

    for iter = 1:10
        order = randperm(size(S_t, 1));
        for sub = 1:size(S_t, 1)
            idx = order(sub);
            row_i = S_t(idx, :);
            c_t = c_t + (u_t(idx) - row_i * c_t) / (row_i * row_i') * row_i';
        end
        c_t(c_t < 0) = 0;
    end

    ima_t = reshape(c_t, 50, 50);
    ima_t_norm = double(ima_t) / max_val;
    psnr_thresh_noisy(t) = psnr(ima_t_norm, ref_norm);
    ssim_thresh_noisy(t) = ssim(ima_t_norm, ref_norm);

    err_t = abs(ima_t - phantom_ref);

    fig = figure('Name',sprintf('3.6: noisy thresh=%g',thresh_factors(t)),'NumberTitle','off','Position',[50 50 1100 380]);
    colormap(gray);
    subplot(1,3,1); imshow(ima_t, [0 max_val]);         colorbar;
                    title(sprintf('Noisy Kaczmarz (thresh=%.0e)', thresh_factors(t))); axis on;
    subplot(1,3,2); imshow(phantom_ref, [0 max_val]);   colorbar;
                    title('Reference m_{ref}');           axis on;
    subplot(1,3,3); imshow(err_t, [0 0.5*max_val]);    colorbar;
                    title('Error Image');                 axis on;
    sgtitle(sprintf('Part 3.6: noisy, thresh=%.0e  |  PSNR=%.1f dB, SSIM=%.4f', ...
            thresh_factors(t), psnr_thresh_noisy(t), ssim_thresh_noisy(t)), 'FontWeight','bold');
    saveas(fig, fullfile(figDir, sprintf('fig_3_6_thresh_%g.png', thresh_factors(t))));
    close(fig);

    fprintf('  Threshold %.0e (noisy): PSNR = %.2f dB, SSIM = %.4f\n', ...
        thresh_factors(t), psnr_thresh_noisy(t), ssim_thresh_noisy(t));
end

fig = figure('Name','3.6 metrics','NumberTitle','off','Position',[50 50 700 520]);
subplot(2,1,1);
semilogx(thresh_factors, psnr_thresh_noisy, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
xlabel('Threshold (max\_norm orani)'); ylabel('PSNR (dB)');
title('PSNR vs Threshold (Noisy)'); grid on;
subplot(2,1,2);
semilogx(thresh_factors, ssim_thresh_noisy, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
xlabel('Threshold (max\_norm orani)'); ylabel('SSIM');
title('SSIM vs Threshold (Noisy)'); grid on;
sgtitle('Part 3.6: Noisy – Image Quality vs Row-norm Threshold', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_3_6_psnr_ssim.png'));

%% 3.7 – Gurultulu Regularized Kaczmarz (Part 3.5 tekrari)

fprintf('\nPart 3.7: Noisy regularized Kaczmarz with different lambda_rel...\n');

psnr_lrel_noisy = zeros(1, N_lrel);
ssim_lrel_noisy = zeros(1, N_lrel);

for t = 1:N_lrel
    lam = sigma_vals(1)^2 * lambda_rel_arr(t);
    c_t = zeros(n_cols_full, 1);

    for iter = 1:10
        order = randperm(n_rows_full);
        for sub = 1:n_rows_full
            idx = order(sub);
            row_i = S(idx, :);
            c_t = c_t + (u_noisy(idx) - row_i * c_t) / (row_i * row_i' + lam) * row_i';
        end
        c_t(c_t < 0) = 0;
    end

    ima_t = reshape(c_t, 50, 50);
    ima_t_norm = double(ima_t) / max_val;
    psnr_lrel_noisy(t) = psnr(ima_t_norm, ref_norm);
    ssim_lrel_noisy(t) = ssim(ima_t_norm, ref_norm);

    err_t = abs(ima_t - phantom_ref);

    fig = figure('Name',sprintf('3.7: noisy lrel=%g',lambda_rel_arr(t)),'NumberTitle','off','Position',[50 50 1100 380]);
    colormap(gray);
    subplot(1,3,1); imshow(ima_t, [0 max_val]);         colorbar;
                    title(sprintf('Noisy Reg. Kacz. (\\lambda_{rel}=%.0e)', lambda_rel_arr(t))); axis on;
    subplot(1,3,2); imshow(phantom_ref, [0 max_val]);   colorbar;
                    title('Reference m_{ref}');           axis on;
    subplot(1,3,3); imshow(err_t, [0 0.5*max_val]);    colorbar;
                    title('Error Image');                 axis on;
    sgtitle(sprintf('Part 3.7: noisy \\lambda_{rel}=%.0e  |  PSNR=%.1f dB, SSIM=%.4f', ...
            lambda_rel_arr(t), psnr_lrel_noisy(t), ssim_lrel_noisy(t)), 'FontWeight','bold');
    saveas(fig, fullfile(figDir, sprintf('fig_3_7_lrel_%g.png', lambda_rel_arr(t))));
    close(fig);

    fprintf('  lambda_rel = %.0e (noisy): PSNR = %.2f dB, SSIM = %.4f\n', ...
        lambda_rel_arr(t), psnr_lrel_noisy(t), ssim_lrel_noisy(t));
end

fig = figure('Name','3.7 metrics','NumberTitle','off','Position',[50 50 700 520]);
subplot(2,1,1);
semilogx(lambda_rel_arr, psnr_lrel_noisy, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b');
xlabel('\lambda_{rel}'); ylabel('PSNR (dB)');
title('PSNR vs \lambda_{rel} (Noisy)'); grid on;
subplot(2,1,2);
semilogx(lambda_rel_arr, ssim_lrel_noisy, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
xlabel('\lambda_{rel}'); ylabel('SSIM');
title('SSIM vs \lambda_{rel} (Noisy)'); grid on;
sgtitle('Part 3.7: Noisy – Image Quality vs \lambda_{rel}', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_3_7_psnr_ssim.png'));

fprintf('\n=== Part III complete. ===\n');

%% 4.1 – Open MPI verisini yukle

fprintf('\nPart 4.1: Loading su_openmpi.mat (may take a while)...\n');
load('su_openmpi.mat');

fprintf('Size of S: %d x %d\n', size(S,1), size(S,2));
fprintf('Size of u: %d x %d\n', size(u,1), size(u,2));

locs_4 = [1, 25000, 50000];
fig = figure('Name','4.1','NumberTitle','off','Position',[50 50 900 600]);
for i = 1:3
    subplot(3,1,i);
    plot(S(:, locs_4(i)), 'LineWidth', 1);
    xlabel('Row Index'); ylabel('Value');
    title(['Column ' num2str(locs_4(i)) ' of S']);
    grid on;
end
sgtitle('Part 4.1: Columns of System Matrix S (OpenMPI)', 'FontWeight','bold');
saveas(fig, fullfile(figDir, 'fig_4_1_S_columns.png'));

%% 4.2 – OpenMPI SVD

fprintf('\nPart 4.2: Computing compact SVD of S (OpenMPI, this will take a while)...\n');
[U4, Sigma4, V4] = svd(S, 'econ');
sigma_vals_4 = diag(Sigma4);

fig = figure('Name','4.2','NumberTitle','off','Position',[50 50 800 400]);
semilogy(sigma_vals_4, 'LineWidth', 1.5);
xlabel('Index'); ylabel('Singular value');
title('Part 4.2 - Singular values of S (OpenMPI)', 'FontWeight','bold');
grid on;
saveas(fig, fullfile(figDir, 'fig_4_2_singular_values.png'));

cond_4 = sigma_vals_4(1) / sigma_vals_4(end);
fprintf('Condition number (OpenMPI): %.4e\n', cond_4);

%% 4.3 – Regularized Kaczmarz (OpenMPI)

lambda_43 = sigma_vals_4(1) * sigma_vals_4(end);
fprintf('\nPart 4.3: Regularized Kaczmarz (lambda = sigma_1*sigma_N = %.4e)\n', lambda_43);

[n_rows_4, n_cols_4] = size(S);
c_4 = zeros(n_cols_4, 1);

for iter = 1:10
    order = randperm(n_rows_4);
    for sub = 1:n_rows_4
        idx = order(sub);
        row_i = S(idx, :);
        c_4 = c_4 + (u(idx) - row_i * c_4) / (row_i * row_i' + lambda_43) * row_i';
    end
    c_4(c_4 < 0) = 0;
    fprintf('  Iter %d done.\n', iter);
end

ima_4 = reshape(c_4, 37, 37, 37);

fig = figure('Name','4.3','NumberTitle','off','Position',[50 50 1000 600]);
montage(reshape(ima_4, [37, 37, 1, 37]), 'DisplayRange', []);
title('Part 4.3: Regularized Kaczmarz – 37 slices (iter 10)', 'FontWeight','bold');
colormap(gray); colorbar;
saveas(fig, fullfile(figDir, 'fig_4_3_montage.png'));

%% 4.4 – Farkli lambda_rel icin Regularized Kaczmarz (OpenMPI)

lambda_rel_4 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
N_lrel4 = length(lambda_rel_4);

fprintf('\nPart 4.4: Regularized Kaczmarz with different lambda_rel (OpenMPI)...\n');

for t = 1:N_lrel4
    lam = sigma_vals_4(1)^2 * lambda_rel_4(t);
    c_t = zeros(n_cols_4, 1);

    for iter = 1:10
        order = randperm(n_rows_4);
        for sub = 1:n_rows_4
            idx = order(sub);
            row_i = S(idx, :);
            c_t = c_t + (u(idx) - row_i * c_t) / (row_i * row_i' + lam) * row_i';
        end
        c_t(c_t < 0) = 0;
    end

    ima_t = reshape(c_t, 37, 37, 37);

    fig = figure('Name',sprintf('4.4: lrel=%g',lambda_rel_4(t)),'NumberTitle','off','Position',[50 50 1000 600]);
    montage(reshape(ima_t, [37, 37, 1, 37]), 'DisplayRange', []);
    title(sprintf('Part 4.4: \\lambda_{rel}=%.0e – 37 slices (iter 10)', lambda_rel_4(t)), 'FontWeight','bold');
    colormap(gray); colorbar;
    saveas(fig, fullfile(figDir, sprintf('fig_4_4_lrel_%g.png', lambda_rel_4(t))));
    close(fig);

    fprintf('  lambda_rel = %.0e done.\n', lambda_rel_4(t));
end

fprintf('\n=== All parts complete. Figures saved to: %s ===\n', figDir);
