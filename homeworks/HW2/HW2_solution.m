%% SETUP
plotDir = fullfile('C:\Users\EKINO\Desktop\ee475\homeworks\HW2','results','plots');
matDir  = fullfile('C:\Users\EKINO\Desktop\ee475\homeworks\HW2','results','mat');
mkdir(plotDir); mkdir(matDir);
normImg = @(im) abs(im)./max(abs(im(:)));
orientGrid = @(im) normImg(im.');
% Make all figures large so imshow images are not tiny when saved
set(0, 'DefaultFigurePosition', [100 100 600 500]);
% Force imshow to scale images to fill the figure (not native pixel size)
iptsetpref('ImshowInitialMagnification', 'fit');

%% 1.1 - Load spiral.mat and plot trajectory
load('spiral.mat');
figure; plot(ktraj); xlabel('k_x'); ylabel('k_y');
title('Interleaved Spiral K-Space Trajectory'); axis equal; grid on;
saveas(gcf, fullfile(plotDir,'1_1_spiral_trajectory.png'));
disp('1.1 done.');

%% 1.2 - Density compensation via Voronoi
area_raw = voronoidens(ktraj);
nan_count = sum(isnan(area_raw(:)));
fprintf('NaN count: %d\n', nan_count);
figure; plot(area_raw(:)); xlabel('Sample index'); ylabel('Area');
title('Raw Voronoi Area'); saveas(gcf, fullfile(plotDir,'1_2_raw_area.png'));
valid = area_raw(~isnan(area_raw(:)) & area_raw(:)<1e-2);
figure; plot(area_raw(:)); ylim([0 max(valid)*1.5]);
xlabel('Sample index'); ylabel('Area'); title('Voronoi Area - Zoomed');
saveas(gcf, fullfile(plotDir,'1_2_raw_area_zoomed.png'));
disp('1.2 done.');

%% 1.3 - Correct density compensation
area_corr = area_raw;
fin = area_raw(isfinite(area_raw(:)));
thr = 3*median(fin);
area_corr(isnan(area_corr)) = thr;
area_corr(area_corr > thr)  = thr;
fprintf('Threshold: %.6f, remaining NaN: %d\n', thr, sum(isnan(area_corr(:))));
figure; plot(area_corr(:)); xlabel('Sample index'); ylabel('Area');
title('Corrected Area'); saveas(gcf, fullfile(plotDir,'1_3_corrected_area.png'));
disp('1.3 done.');

%% 1.4 - Direct Summation (reference, N=128, dx=1)
N=128; dx=1; tau=N/2;
kx_vec=real(ktraj(:)); ky_vec=imag(ktraj(:));
M_vec=kdata(:); d_vec=area_corr(:);
dM = d_vec.*M_vec;
n_vals = (0:N-1)-tau;
phase_y = exp(1j*2*pi*dx * ky_vec * n_vals);
ima_direct = zeros(N,N);
t0 = cputime;
for ix = 1:N
    phase_x = exp(1j*2*pi*dx * kx_vec * n_vals(ix));
    ima_direct(ix,:) = (dM.*phase_x).' * phase_y;
end
t_direct = cputime-t0;
clear phase_y phase_x;
ima_direct = normImg(ima_direct.');
fprintf('1.4 CPU time: %.2f s\n', t_direct);
figure; imshow(ima_direct,[0 1]); colormap gray; colorbar;
title('Direct Summation 128x128'); saveas(gcf, fullfile(plotDir,'1_4_image.png'));
figure; plot(abs(ima_direct(end/2,:))); xlabel('Column'); ylabel('Intensity');
title('1.4 Horizontal Cross-Section'); grid on; saveas(gcf, fullfile(plotDir,'1_4_horiz.png'));
figure; plot(abs(ima_direct(:,end/2))); xlabel('Row'); ylabel('Intensity');
title('1.4 Vertical Cross-Section'); grid on; saveas(gcf, fullfile(plotDir,'1_4_vert.png'));
save(fullfile(matDir,'ima_direct.mat'),'ima_direct');
disp('1.4 done.');

%% 1.5 - No Density Compensation
n_vals = (0:N-1)-tau;
phase_y = exp(1j*2*pi*dx * ky_vec * n_vals);
ima_nodcf = zeros(N,N);
t0 = cputime;
for ix = 1:N
    phase_x = exp(1j*2*pi*dx * kx_vec * n_vals(ix));
    ima_nodcf(ix,:) = (M_vec.*phase_x).' * phase_y;
end
t_nodcf = cputime-t0; clear phase_y phase_x;
ima_nodcf = normImg(ima_nodcf.');
err_nodcf = abs(ima_nodcf - ima_direct);
psnr_nodcf = psnr(ima_nodcf, ima_direct);
ssim_nodcf = ssim(ima_nodcf, ima_direct);
fprintf('1.5 No DCF: PSNR=%.2f dB, SSIM=%.4f, CPU=%.2f s\n', psnr_nodcf, ssim_nodcf, t_nodcf);
figure; imshow(ima_nodcf,[0 1]); colormap gray; colorbar;
title('1.5 No DCF Image'); saveas(gcf, fullfile(plotDir,'1_5_image.png'));
figure; imshow(err_nodcf,[0 0.2]); colormap gray; colorbar;
title('1.5 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'1_5_error.png'));
figure; plot(abs(ima_nodcf(end/2,:)),'b'); hold on;
plot(abs(ima_direct(end/2,:)),'r--'); legend('NoDCF','Ref');
xlabel('Column'); title('1.5 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'1_5_horiz.png'));
figure; plot(abs(ima_nodcf(:,end/2)),'b'); hold on;
plot(abs(ima_direct(:,end/2)),'r--'); legend('NoDCF','Ref');
xlabel('Row'); title('1.5 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'1_5_vert.png'));
disp('1.5 done.');

%% 1.6 - Double FOV (N=256, dx=1)
N2=256; tau2=N2/2; n2=(0:N2-1)-tau2;
phase_y2 = exp(1j*2*pi*dx * ky_vec * n2);
ima_dfov = zeros(N2,N2);
t0 = cputime;
for ix = 1:N2
    px = exp(1j*2*pi*dx * kx_vec * n2(ix));
    ima_dfov(ix,:) = (dM.*px).' * phase_y2;
end
t_dfov = cputime-t0; clear phase_y2 px;
ima_dfov = normImg(ima_dfov.');
c=N2/2; h=N/2;
ima_dfov_crop = normImg(ima_dfov(c-h+1:c+h, c-h+1:c+h));
err_dfov = abs(ima_dfov_crop - ima_direct);
psnr_dfov=psnr(ima_dfov_crop,ima_direct); ssim_dfov=ssim(ima_dfov_crop,ima_direct);
fprintf('1.6 DblFOV: PSNR=%.2f, SSIM=%.4f, CPU=%.2f s\n',psnr_dfov,ssim_dfov,t_dfov);
figure; imshow(ima_dfov,[0 1]); colormap gray; colorbar;
title('1.6 Full 256x256'); saveas(gcf, fullfile(plotDir,'1_6_full.png'));
figure; imshow(ima_dfov_crop,[0 1]); colormap gray; colorbar;
title('1.6 Cropped 128x128'); saveas(gcf, fullfile(plotDir,'1_6_crop.png'));
figure; imshow(err_dfov,[0 0.2]); colormap gray; colorbar;
title('1.6 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'1_6_error.png'));
figure; plot(abs(ima_dfov_crop(end/2,:)),'b'); hold on;
plot(abs(ima_direct(end/2,:)),'r--'); legend('DblFOV','Ref');
title('1.6 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'1_6_horiz.png'));
figure; plot(abs(ima_dfov_crop(:,end/2)),'b'); hold on;
plot(abs(ima_direct(:,end/2)),'r--'); legend('DblFOV','Ref');
title('1.6 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'1_6_vert.png'));
disp('1.6 done.');

%% 1.7 - Half Pixel Size (dx=0.5, N=256)
N3=256; dx3=0.5; tau3=N3/2; n3=(0:N3-1)-tau3;
phase_y3 = exp(1j*2*pi*dx3 * ky_vec * n3);
ima_hpx = zeros(N3,N3);
t0=cputime;
for ix=1:N3
    px=exp(1j*2*pi*dx3*kx_vec*n3(ix));
    ima_hpx(ix,:)=(dM.*px).'*phase_y3;
end
t_hpx=cputime-t0; clear phase_y3 px;
ima_hpx = normImg(ima_hpx.');
kref=fftshift(fft2(ifftshift(ima_direct)));
pad=(N3-N)/2; kref_pad=zeros(N3,N3);
kref_pad(pad+1:pad+N, pad+1:pad+N)=kref;
ima_ref256=normImg(fftshift(ifft2(ifftshift(kref_pad))));
err_hpx=abs(ima_hpx-ima_ref256);
psnr_hpx=psnr(ima_hpx,ima_ref256); ssim_hpx=ssim(ima_hpx,ima_ref256);
fprintf('1.7 HalfPx: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_hpx,ssim_hpx,t_hpx);
figure; imshow(ima_hpx,[0 1]); colormap gray; colorbar;
title('1.7 Half Pixel 256x256'); saveas(gcf, fullfile(plotDir,'1_7_image.png'));
figure; imshow(ima_ref256,[0 1]); colormap gray; colorbar;
title('1.7 Ref ZP 256x256'); saveas(gcf, fullfile(plotDir,'1_7_ref.png'));
figure; imshow(err_hpx,[0 0.2]); colormap gray; colorbar;
title('1.7 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'1_7_error.png'));
figure; plot(abs(ima_hpx(end/2,:)),'b'); hold on;
plot(abs(ima_ref256(end/2,:)),'r--'); legend('HalfPx','Ref');
title('1.7 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'1_7_horiz.png'));
figure; plot(abs(ima_hpx(:,end/2)),'b'); hold on;
plot(abs(ima_ref256(:,end/2)),'r--'); legend('HalfPx','Ref');
title('1.7 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'1_7_vert.png'));
disp('1.7 done.');

%% 1.8 - Double Pixel Size (dx=2, N=64)
N4=64; dx4=2; tau4=N4/2; n4=(0:N4-1)-tau4;
phase_y4=exp(1j*2*pi*dx4*ky_vec*n4);
ima_dpx=zeros(N4,N4);
t0=cputime;
for ix=1:N4
    px=exp(1j*2*pi*dx4*kx_vec*n4(ix));
    ima_dpx(ix,:)=(dM.*px).'*phase_y4;
end
t_dpx=cputime-t0; clear phase_y4 px;
ima_dpx=normImg(ima_dpx.');
kdpx=fftshift(fft2(ifftshift(ima_dpx)));
pad2=(N-N4)/2; kdpx_pad=zeros(N,N);
kdpx_pad(pad2+1:pad2+N4,pad2+1:pad2+N4)=kdpx;
ima_dpx_zp=normImg(fftshift(ifft2(ifftshift(kdpx_pad))));
err_dpx=abs(ima_dpx_zp-ima_direct);
psnr_dpx=psnr(ima_dpx_zp,ima_direct); ssim_dpx=ssim(ima_dpx_zp,ima_direct);
fprintf('1.8 DblPx: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_dpx,ssim_dpx,t_dpx);
figure; imshow(ima_dpx,[0 1]); colormap gray; colorbar;
title('1.8 Original 64x64'); saveas(gcf, fullfile(plotDir,'1_8_image.png'));
figure; imshow(ima_dpx_zp,[0 1]); colormap gray; colorbar;
title('1.8 Zero-padded 128x128'); saveas(gcf, fullfile(plotDir,'1_8_zp.png'));
figure; imshow(err_dpx,[0 0.2]); colormap gray; colorbar;
title('1.8 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'1_8_error.png'));
figure; plot(abs(ima_dpx_zp(end/2,:)),'b'); hold on;
plot(abs(ima_direct(end/2,:)),'r--'); legend('DblPx','Ref');
title('1.8 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'1_8_horiz.png'));
figure; plot(abs(ima_dpx_zp(:,end/2)),'b'); hold on;
plot(abs(ima_direct(:,end/2)),'r--'); legend('DblPx','Ref');
title('1.8 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'1_8_vert.png'));
disp('1.8 done.');

%% 1.9 - 1X Gridding (osf=1, wg=2)
t0=cputime;
ima_g1=gridkb(kdata,ktraj,area_corr,128,1,2,'image');
t_g1=cputime-t0;
% Match the image-axis convention used by the direct-summation reference
ima_g1=orientGrid(ima_g1);
err_g1=abs(ima_g1-ima_direct);
psnr_g1=psnr(ima_g1,ima_direct); ssim_g1=ssim(ima_g1,ima_direct);
fprintf('1.9 Grid1X: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_g1,ssim_g1,t_g1);
figure; imshow(ima_g1,[0 1]); colormap gray; colorbar;
title('1.9 Gridding 1X'); saveas(gcf, fullfile(plotDir,'1_9_image.png'));
figure; imshow(err_g1,[0 0.2]); colormap gray; colorbar;
title('1.9 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'1_9_error.png'));
figure; plot(abs(ima_g1(end/2,:)),'b'); hold on;
plot(abs(ima_direct(end/2,:)),'r--'); legend('1XGrid','Ref');
title('1.9 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'1_9_horiz.png'));
figure; plot(abs(ima_g1(:,end/2)),'b'); hold on;
plot(abs(ima_direct(:,end/2)),'r--'); legend('1XGrid','Ref');
title('1.9 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'1_9_vert.png'));
disp('1.9 done.');

%% 1.10 - 2X Gridding (osf=2, wg=4)
t0=cputime;
ima_g2_full=gridkb(kdata,ktraj,area_corr,128,2,4,'image');
t_g2=cputime-t0;
ima_g2_full=orientGrid(ima_g2_full);
c2=128; h2=64;
ima_g2=normImg(ima_g2_full(c2-h2+1:c2+h2,c2-h2+1:c2+h2));
err_g2=abs(ima_g2-ima_direct);
psnr_g2=psnr(ima_g2,ima_direct); ssim_g2=ssim(ima_g2,ima_direct);
fprintf('1.10 Grid2X: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_g2,ssim_g2,t_g2);
figure; imshow(ima_g2_full,[0 1]); colormap gray; colorbar;
title('1.10 Full 256x256'); saveas(gcf, fullfile(plotDir,'1_10_full.png'));
figure; imshow(ima_g2,[0 1]); colormap gray; colorbar;
title('1.10 Cropped 128x128'); saveas(gcf, fullfile(plotDir,'1_10_crop.png'));
figure; imshow(err_g2,[0 0.2]); colormap gray; colorbar;
title('1.10 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'1_10_error.png'));
figure; plot(abs(ima_g2(end/2,:)),'b'); hold on;
plot(abs(ima_direct(end/2,:)),'r--'); legend('2XGrid','Ref');
title('1.10 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'1_10_horiz.png'));
figure; plot(abs(ima_g2(:,end/2)),'b'); hold on;
plot(abs(ima_direct(:,end/2)),'r--'); legend('2XGrid','Ref');
title('1.10 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'1_10_vert.png'));
disp('1.10 done.');

%% 1.11 - Scattered Interpolation 1X (128x128)
kx_s=real(ktraj(:)); ky_s=imag(ktraj(:));
F_re = scatteredInterpolant(kx_s,ky_s,real(kdata(:)),'linear','none');
F_im = scatteredInterpolant(kx_s,ky_s,imag(kdata(:)),'linear','none');
kg1d = linspace(-0.5,0.5,128);
[KX1,KY1] = meshgrid(kg1d,kg1d);
t0=cputime;
kgrid1 = F_re(KX1,KY1) + 1j*F_im(KX1,KY1);
t_s1=cputime-t0;
kgrid1(isnan(kgrid1))=0;
ima_s1 = fftshift(ifft2(ifftshift(kgrid1)));
ima_s1 = normImg(ima_s1);
candidates = {ima_s1, fliplr(ima_s1), flipud(ima_s1), ima_s1.', flipud(fliplr(ima_s1)), rot90(ima_s1,1), rot90(ima_s1,3)};
best_p=-Inf;
for ci=1:numel(candidates)
    if all(size(candidates{ci})==[N N])
        pp=psnr(candidates{ci},ima_direct);
        if pp>best_p, best_p=pp; ima_s1=candidates{ci}; end
    end
end
err_s1=abs(ima_s1-ima_direct);
psnr_s1=psnr(ima_s1,ima_direct); ssim_s1=ssim(ima_s1,ima_direct);
fprintf('1.11 ScatInterp1X: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_s1,ssim_s1,t_s1);
figure; imshow(ima_s1,[0 1]); colormap gray; colorbar;
title('1.11 ScatInterp 1X'); saveas(gcf, fullfile(plotDir,'1_11_image.png'));
figure; imshow(err_s1,[0 0.2]); colormap gray; colorbar;
title('1.11 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'1_11_error.png'));
figure; plot(abs(ima_s1(end/2,:)),'b'); hold on;
plot(abs(ima_direct(end/2,:)),'r--'); legend('Scat1X','Ref');
title('1.11 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'1_11_horiz.png'));
figure; plot(abs(ima_s1(:,end/2)),'b'); hold on;
plot(abs(ima_direct(:,end/2)),'r--'); legend('Scat1X','Ref');
title('1.11 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'1_11_vert.png'));
disp('1.11 done.');

%% 1.12 - Scattered Interpolation 2X (256x256)
kg2d = linspace(-0.5,0.5,256);
[KX2,KY2] = meshgrid(kg2d,kg2d);
t0=cputime;
kgrid2 = F_re(KX2,KY2) + 1j*F_im(KX2,KY2);
t_s2=cputime-t0;
kgrid2(isnan(kgrid2))=0;
ima_s2_full = fftshift(ifft2(ifftshift(kgrid2)));
ima_s2_full = normImg(ima_s2_full);
candidates2 = {ima_s2_full, fliplr(ima_s2_full), flipud(ima_s2_full), ima_s2_full.', flipud(fliplr(ima_s2_full)), rot90(ima_s2_full,1), rot90(ima_s2_full,3)};
best_p=-Inf;
for ci=1:numel(candidates2)
    tmp = candidates2{ci};
    if all(size(tmp)==[256 256])
        c3=128;
        crop=normImg(tmp(c3-h2+1:c3+h2,c3-h2+1:c3+h2));
        pp=psnr(crop,ima_direct);
        if pp>best_p, best_p=pp; ima_s2_full=tmp; ima_s2=crop; end
    end
end
err_s2=abs(ima_s2-ima_direct);
psnr_s2=psnr(ima_s2,ima_direct); ssim_s2=ssim(ima_s2,ima_direct);
fprintf('1.12 ScatInterp2X: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_s2,ssim_s2,t_s2);
figure; imshow(ima_s2_full,[0 1]); colormap gray; colorbar;
title('1.12 Full 256x256'); saveas(gcf, fullfile(plotDir,'1_12_full.png'));
figure; imshow(ima_s2,[0 1]); colormap gray; colorbar;
title('1.12 Cropped 128x128'); saveas(gcf, fullfile(plotDir,'1_12_crop.png'));
figure; imshow(err_s2,[0 0.2]); colormap gray; colorbar;
title('1.12 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'1_12_error.png'));
figure; plot(abs(ima_s2(end/2,:)),'b'); hold on;
plot(abs(ima_direct(end/2,:)),'r--'); legend('Scat2X','Ref');
title('1.12 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'1_12_horiz.png'));
figure; plot(abs(ima_s2(:,end/2)),'b'); hold on;
plot(abs(ima_direct(:,end/2)),'r--'); legend('Scat2X','Ref');
title('1.12 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'1_12_vert.png'));
disp('1.12 done.');

%% ========== PART II ==========

%% 2.1 - Phantom and sinogram
P = phantom('Modified Shepp-Logan',256);
P = normImg(P);
angles = 0:179;
proj = radon(P, angles);
figure; imshow(P,[0 1]); colormap gray; colorbar;
title('Shepp-Logan Phantom 256x256'); saveas(gcf, fullfile(plotDir,'2_1_phantom.png'));
figure; imagesc(angles,1:size(proj,1),proj); colormap gray; colorbar;
xlabel('\theta (degrees)'); ylabel('l'); title('Sinogram'); axis xy;
saveas(gcf, fullfile(plotDir,'2_1_sinogram.png'));
figure; plot(P(128,:)); xlabel('Column'); title('2.1 Phantom Horiz XS'); grid on;
saveas(gcf, fullfile(plotDir,'2_1_phantom_horiz.png'));
figure; plot(P(:,128)); xlabel('Row'); title('2.1 Phantom Vert XS'); grid on;
saveas(gcf, fullfile(plotDir,'2_1_phantom_vert.png'));
disp('2.1 done.');

%% 2.2 - Filtered Backprojection (Ram-Lak)
t0=cputime;
ima_fbp_raw = iradon(proj, angles);
t_fbp=cputime-t0;
csz=round(size(ima_fbp_raw,1)/2);
ima_fbp=ima_fbp_raw(csz-127:csz+128, csz-127:csz+128);
ima_fbp=normImg(ima_fbp);
err_fbp=abs(ima_fbp-P);
psnr_fbp=psnr(ima_fbp,P); ssim_fbp=ssim(ima_fbp,P);
fprintf('2.2 FBP: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_fbp,ssim_fbp,t_fbp);
figure; imshow(ima_fbp,[0 1]); colormap gray; colorbar;
title('2.2 FBP Image'); saveas(gcf, fullfile(plotDir,'2_2_image.png'));
figure; imshow(err_fbp,[0 0.2]); colormap gray; colorbar;
title('2.2 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'2_2_error.png'));
figure; plot(ima_fbp(128,:),'b'); hold on; plot(P(128,:),'r--');
legend('FBP','Ref'); title('2.2 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'2_2_horiz.png'));
figure; plot(ima_fbp(:,128),'b'); hold on; plot(P(:,128),'r--');
legend('FBP','Ref'); title('2.2 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'2_2_vert.png'));
disp('2.2 done.');

%% 2.3 - Naive Backprojection (no filter)
t0=cputime;
ima_bp_raw = iradon(proj, angles, 'linear', 'none');
t_bp=cputime-t0;
ima_bp=ima_bp_raw(csz-127:csz+128, csz-127:csz+128);
ima_bp=normImg(ima_bp);
err_bp=abs(ima_bp-P);
psnr_bp=psnr(ima_bp,P); ssim_bp=ssim(ima_bp,P);
fprintf('2.3 NaiveBP: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_bp,ssim_bp,t_bp);
figure; imshow(ima_bp,[0 1]); colormap gray; colorbar;
title('2.3 Naive BP'); saveas(gcf, fullfile(plotDir,'2_3_image.png'));
figure; imshow(err_bp,[0 0.2]); colormap gray; colorbar;
title('2.3 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'2_3_error.png'));
figure; plot(ima_bp(128,:),'b'); hold on; plot(P(128,:),'r--');
legend('NaiveBP','Ref'); title('2.3 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'2_3_horiz.png'));
figure; plot(ima_bp(:,128),'b'); hold on; plot(P(:,128),'r--');
legend('NaiveBP','Ref'); title('2.3 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'2_3_vert.png'));
disp('2.3 done.');

%% 2.4 - Projection-Slice Theorem
Np = size(proj,1); kmax = 0.5;
kr_vec = linspace(-kmax, kmax, Np).';
kdata_proj = zeros(Np,180);
for th=1:180
    kdata_proj(:,th) = fftshift(fft(ifftshift(proj(:,th))));
end
theta_rad = deg2rad(angles);
kx_proj = kr_vec * cos(theta_rad);
ky_proj = kr_vec * sin(theta_rad);
ktraj_proj = kx_proj + 1j*ky_proj;
figure; plot(real(ktraj_proj(:)),imag(ktraj_proj(:)),'b.','MarkerSize',1);
xlabel('k_x'); ylabel('k_y'); title('2.4 Radial Trajectory'); axis equal; grid on;
saveas(gcf, fullfile(plotDir,'2_4_trajectory.png'));
figure; imagesc(angles, kr_vec, log(1+abs(kdata_proj))); colormap gray; colorbar;
xlabel('\theta'); ylabel('k_r'); title('2.4 K-Space Magnitude (log)'); axis xy;
saveas(gcf, fullfile(plotDir,'2_4_kspace_mag.png'));
disp('2.4 done.');

%% 2.5 - Density compensation (geometric, radial)
dkr = 2*kmax/(Np-1); n_ang = 180;
area_proj = abs(kr_vec)*(2*pi/n_ang)*dkr;
[~,idx_dc]=min(abs(kr_vec));
area_proj(idx_dc) = pi*(dkr/2)^2 / n_ang;
area_proj2d = repmat(area_proj, 1, n_ang);
figure; plot(kr_vec, area_proj); xlabel('k_r'); ylabel('Area');
title('2.5 DCF (first angle)'); grid on; saveas(gcf, fullfile(plotDir,'2_5_dcf.png'));
disp('2.5 done.');

%% 2.6 - Direct Summation (projection data, 256x256)
Np2=256; dxp=1; taup=Np2/2;
kxp=real(ktraj_proj(:)); kyp=imag(ktraj_proj(:));
dMp=area_proj2d(:).*kdata_proj(:);
np=(0:Np2-1)-taup;
ima_dsp=zeros(Np2,Np2);
t0=cputime;
for ix=1:Np2
    phx=exp(1j*2*pi*dxp*kxp*np(ix));
    phy=exp(1j*2*pi*dxp*kyp*np);
    ima_dsp(ix,:)=(dMp.*phx).'*phy;
    if mod(ix,64)==0, fprintf('  2.6 row %d/%d\n',ix,Np2); end
end
t_dsp=cputime-t0; clear phx phy;
ima_dsp=normImg(ima_dsp.');
cands={ima_dsp,flipud(ima_dsp),fliplr(ima_dsp),ima_dsp.',flipud(fliplr(ima_dsp)),rot90(ima_dsp,1),rot90(ima_dsp,3)};
bp=-Inf;
for ci=1:numel(cands)
    if all(size(cands{ci})==[256 256])
        pp=psnr(cands{ci},P); if pp>bp, bp=pp; ima_dsp=cands{ci}; end
    end
end
err_dsp=abs(ima_dsp-P);
psnr_dsp=psnr(ima_dsp,P); ssim_dsp=ssim(ima_dsp,P);
fprintf('2.6 DS: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_dsp,ssim_dsp,t_dsp);
figure; imshow(ima_dsp,[0 1]); colormap gray; colorbar;
title('2.6 DS Radial'); saveas(gcf, fullfile(plotDir,'2_6_image.png'));
figure; imshow(err_dsp,[0 0.2]); colormap gray; colorbar;
title('2.6 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'2_6_error.png'));
figure; plot(ima_dsp(128,:),'b'); hold on; plot(P(128,:),'r--');
legend('DS','Ref'); title('2.6 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'2_6_horiz.png'));
figure; plot(ima_dsp(:,128),'b'); hold on; plot(P(:,128),'r--');
legend('DS','Ref'); title('2.6 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'2_6_vert.png'));
disp('2.6 done.');

%% 2.7 - 2X Gridding (projection data)
t0=cputime;
ima_g2p_full=gridkb(kdata_proj,ktraj_proj,area_proj2d,256,2,4,'image');
t_g2p=cputime-t0;
% For projection-data gridding, transpose + vertical flip matches phantom orientation
ima_g2p_full=normImg(rot90(ima_g2p_full,1));
c5=256; h5=128;
ima_g2p=normImg(ima_g2p_full(c5-h5+1:c5+h5,c5-h5+1:c5+h5));
err_g2p=abs(ima_g2p-P);
psnr_g2p=psnr(ima_g2p,P); ssim_g2p=ssim(ima_g2p,P);
fprintf('2.7 Grid2X: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_g2p,ssim_g2p,t_g2p);
figure; imshow(ima_g2p_full,[0 1]); colormap gray; colorbar;
title('2.7 Full Gridding Output'); saveas(gcf, fullfile(plotDir,'2_7_full.png'));
figure; imshow(ima_g2p,[0 1]); colormap gray; colorbar;
title('2.7 Cropped 256x256'); saveas(gcf, fullfile(plotDir,'2_7_crop.png'));
figure; imshow(err_g2p,[0 0.2]); colormap gray; colorbar;
title('2.7 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'2_7_error.png'));
figure; plot(ima_g2p(128,:),'b'); hold on; plot(P(128,:),'r--');
legend('Grid2X','Ref'); title('2.7 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'2_7_horiz.png'));
figure; plot(ima_g2p(:,128),'b'); hold on; plot(P(:,128),'r--');
legend('Grid2X','Ref'); title('2.7 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'2_7_vert.png'));
disp('2.7 done.');

%% ========== PART III ==========

%% 3.1 - Load radial data and display trajectory
load('shepplogan_radial_data.mat');
figure; plot(real(ktraj(:)),imag(ktraj(:)),'b.','MarkerSize',1);
xlabel('k_x'); ylabel('k_y'); title('3.1 Radial Trajectory (one-sided)');
axis equal; grid on; saveas(gcf, fullfile(plotDir,'3_1_trajectory.png'));


%% 3.2
[Ns,Nl]=size(ktraj); kmax_r=max(abs(ktraj(:)));
kr_r=abs(ktraj(:,1)); dkr_r=kmax_r/(Ns-1);
area_rad1d=kr_r*(2*pi/Nl)*dkr_r;
area_rad1d(1)=pi*(dkr_r/2)^2/Nl;
area_radial=repmat(area_rad1d,1,Nl);
figure; plot(kr_r,area_rad1d); xlabel('k_r'); ylabel('Area');
title('3.2 DCF (one radial line)'); grid on; saveas(gcf, fullfile(plotDir,'3_2_dcf.png'));


%% 3.3
Nr=256; dxr=1; taur=Nr/2;
kxr=real(ktraj(:)); kyr=imag(ktraj(:));
dMr=area_radial(:).*kdata(:);
nr=(0:Nr-1)-taur;
ima_dsr=zeros(Nr,Nr);
t0=cputime;
for ix=1:Nr
    phx=exp(1j*2*pi*dxr*kxr*nr(ix));
    phy=exp(1j*2*pi*dxr*kyr*nr);
    ima_dsr(ix,:)=(dMr.*phx).'*phy;
    if mod(ix,64)==0, fprintf('  3.3 row %d/%d\n',ix,Nr); end
end
t_dsr=cputime-t0; clear phx phy;
ima_dsr=normImg(ima_dsr.');
cands={ima_dsr,flipud(ima_dsr),fliplr(ima_dsr),ima_dsr.',flipud(fliplr(ima_dsr)),rot90(ima_dsr,1),rot90(ima_dsr,3)};
bp=-Inf;
for ci=1:numel(cands)
    if all(size(cands{ci})==[256 256])
        pp=psnr(cands{ci},P); if pp>bp, bp=pp; ima_dsr=cands{ci}; end
    end
end
err_dsr=abs(ima_dsr-P);
psnr_dsr=psnr(ima_dsr,P); ssim_dsr=ssim(ima_dsr,P);
fprintf('3.3 DS: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_dsr,ssim_dsr,t_dsr);
figure; imshow(ima_dsr,[0 1]); colormap gray; colorbar;
title('3.3 DS Radial'); saveas(gcf, fullfile(plotDir,'3_3_image.png'));
figure; imshow(err_dsr,[0 0.2]); colormap gray; colorbar;
title('3.3 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'3_3_error.png'));
figure; plot(ima_dsr(128,:),'b'); hold on; plot(P(128,:),'r--');
legend('DS','Ref'); title('3.3 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'3_3_horiz.png'));
figure; plot(ima_dsr(:,128),'b'); hold on; plot(P(:,128),'r--');
legend('DS','Ref'); title('3.3 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'3_3_vert.png'));

%% 3.4
t0=cputime;
ima_g2r_full=gridkb(kdata,ktraj,area_radial,256,2,4,'image');
t_g2r=cputime-t0;
% For one-sided radial gridding, transpose + vertical flip matches phantom orientation
ima_g2r_full=normImg(rot90(ima_g2r_full,1));
c6=256;
ima_g2r=normImg(ima_g2r_full(c6-h5+1:c6+h5,c6-h5+1:c6+h5));
err_g2r=abs(ima_g2r-P);
psnr_g2r=psnr(ima_g2r,P); ssim_g2r=ssim(ima_g2r,P);
fprintf('3.4 Grid2X: PSNR=%.2f, SSIM=%.4f, CPU=%.2f\n',psnr_g2r,ssim_g2r,t_g2r);
figure; imshow(ima_g2r_full,[0 1]); colormap gray; colorbar;
title('3.4 Full Gridding Output'); saveas(gcf, fullfile(plotDir,'3_4_full.png'));
figure; imshow(ima_g2r,[0 1]); colormap gray; colorbar;
title('3.4 Cropped 256x256'); saveas(gcf, fullfile(plotDir,'3_4_crop.png'));
figure; imshow(err_g2r,[0 0.2]); colormap gray; colorbar;
title('3.4 Error [0 0.2]'); saveas(gcf, fullfile(plotDir,'3_4_error.png'));
figure; plot(ima_g2r(128,:),'b'); hold on; plot(P(128,:),'r--');
legend('Grid2X','Ref'); title('3.4 Horizontal XS'); grid on; saveas(gcf, fullfile(plotDir,'3_4_horiz.png'));
figure; plot(ima_g2r(:,128),'b'); hold on; plot(P(:,128),'r--');
legend('Grid2X','Ref'); title('3.4 Vertical XS'); grid on; saveas(gcf, fullfile(plotDir,'3_4_vert.png'));

%% SUMMARY TABLE
fprintf('\n===== SUMMARY =====\n');
fprintf('%-30s %8s %8s %8s\n','Method','PSNR','SSIM','CPU(s)');
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.4 Direct Sum (ref)',Inf,1,t_direct);
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.5 No DCF',psnr_nodcf,ssim_nodcf,t_nodcf);
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.6 Double FOV',psnr_dfov,ssim_dfov,t_dfov);
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.7 Half Pixel',psnr_hpx,ssim_hpx,t_hpx);
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.8 Double Pixel',psnr_dpx,ssim_dpx,t_dpx);
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.9 Grid 1X',psnr_g1,ssim_g1,t_g1);
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.10 Grid 2X',psnr_g2,ssim_g2,t_g2);
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.11 ScatInterp 1X',psnr_s1,ssim_s1,t_s1);
fprintf('%-30s %8.2f %8.4f %8.2f\n','1.12 ScatInterp 2X',psnr_s2,ssim_s2,t_s2);
fprintf('%-30s %8.2f %8.4f %8.2f\n','2.2 FBP',psnr_fbp,ssim_fbp,t_fbp);
fprintf('%-30s %8.2f %8.4f %8.2f\n','2.3 Naive BP',psnr_bp,ssim_bp,t_bp);
fprintf('%-30s %8.2f %8.4f %8.2f\n','2.6 DS Proj',psnr_dsp,ssim_dsp,t_dsp);
fprintf('%-30s %8.2f %8.4f %8.2f\n','2.7 Grid Proj',psnr_g2p,ssim_g2p,t_g2p);
fprintf('%-30s %8.2f %8.4f %8.2f\n','3.3 DS Radial',psnr_dsr,ssim_dsr,t_dsr);
fprintf('%-30s %8.2f %8.4f %8.2f\n','3.4 Grid Radial',psnr_g2r,ssim_g2r,t_g2r);
