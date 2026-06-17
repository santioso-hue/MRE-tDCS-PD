% run_qti_cov_cohort.m — QTI covariance fit for the COHORT (md-dmri) -> oriented mean tensor <D>.
%
% Input contract: the cohort (MUDI_synb0/<subj>/) delivers the two b-tensor encodings as SEPARATE
% Synb0+topup+eddy-corrected, UN-smoothed series with bval/bvec (not a pre-combined, pre-smoothed
% series), so this fit builds the xps + merge + smoothing itself:
%   linear_corrected.{nii.gz,bval,bvec}    = LTE  -> b_delta = +1
%   spherical_corrected.{nii.gz,bval,bvec} = STE  -> b_delta =  0   (isotropic; directions unused)
%
% So we (1) build the xps per series tagging the b-tensor shape, (2) merge LTE+STE in matched order,
% (3) elastix step2 inter-series motion-correction (co-align LTE<->STE; REQUIRES elastix on PATH),
% (4) reproduce Christoffer's PD-paper covariance fit: smooth (filter_sigma=0.7) -> auto-mask ->
% data2fit -> fit2param. Output names match our downstream (prepare_dmri_tensor.py):
%   cov_mfs.mat  (<D> = m(:,:,:,2:7))   cov_dps.mat  (MD, uFA, eigenvector u)
%
% Authoritative source (references/ParkMREPipeline, chrol_pipeline):
%   mudi_synb0plusmc_v3.m  -> b_deltas=[1 0], mdm_s_merge({s_lte,s_ste}, ...)
%   step4_fit_model_*.m, branch 'covariance_CO' ("This one was run for PD paper"): filter_sigma=0.7
% The cohort input is UN-smoothed, so we apply filter_sigma=0.7 here (do not skip the smoothing step).
%
% Usage (paths from env; see run_qti_cov_cohort.sh):
%   DMRI_IN=<dir with linear_/spherical_corrected> FIT_OUT=<out dir> MDDMRI_DIR=<md-dmri> \
%   matlab -batch "run('pipeline/run_qti_cov_cohort.m')"

dmri_in = getenv('DMRI_IN');   if isempty(dmri_in), error('Set DMRI_IN');   end
fit_out = getenv('FIT_OUT');   if isempty(fit_out), error('Set FIT_OUT');   end
mddmri  = getenv('MDDMRI_DIR'); if isempty(mddmri), error('Set MDDMRI_DIR'); end
run(fullfile(mddmri, 'setup_paths.m'));

op = fullfile(fit_out, 'qti_cov');
if ~exist(op, 'dir'), mkdir(op); end

lin = fullfile(dmri_in, 'linear_corrected.nii.gz');     % LTE
sph = fullfile(dmri_in, 'spherical_corrected.nii.gz');  % STE

opt = dtd_covariance_opt(mdm_opt());
opt.do_overwrite = 1;
opt.filter_sigma = 0.7;     % chrol step4 'covariance_CO' (PD paper); cohort input is un-smoothed

% --- per-series xps, tagging the b-tensor shape (CRITICAL: LTE=+1, STE=0) ---
[lb, lv] = mdm_fn_nii2bvalbvec(lin);
[sb, sv] = mdm_fn_nii2bvalbvec(sph);
s_lte.nii_fn = lin; s_lte.xps = mdm_xps_from_bval_bvec(lb, lv, 1.0);
s_ste.nii_fn = sph; s_ste.xps = mdm_xps_from_bval_bvec(sb, sv, 0.0);

% --- merge LTE then STE (writes LTE_STE.nii.gz + LTE_STE_xps.mat; returns merged s) ---
s = mdm_s_merge({s_lte, s_ste}, op, 'LTE_STE', opt);

% --- HARD order-alignment guard (the bug the r~0.99 gate exists to catch) ---
% LTE block first, STE block second. Note: b=0 volumes get b_delta=0 by md-dmri convention
% (a zero b-tensor has no anisotropy), so we require linear only on the b>0 LTE volumes.
n_lte = s_lte.xps.n; bd = s.xps.b_delta; bb = s.xps.b;
lte = 1:n_lte; ste = (n_lte+1):s.xps.n;
assert(s.xps.n == s_lte.xps.n + s_ste.xps.n, 'xps.n %d != LTE %d + STE %d', s.xps.n, s_lte.xps.n, s_ste.xps.n);
assert(all(abs(bd(lte(bb(lte) > 0)) - 1) < 1e-6), 'LTE block has a b>0 volume that is not linear (b_delta~=1)');
assert(all(abs(bd(ste) - 0) < 1e-6), 'STE block has a non-spherical volume (b_delta~=0) -- LTE/STE order broken');
fprintf('Merged %d vols (LTE %d + STE %d); b %.0f..%.0f s/mm^2; LTE(b>0)=linear, STE=spherical verified.\n', ...
    s.xps.n, n_lte, s_ste.xps.n, min(bb)/1e6, max(bb)/1e6);

% Fail fast on a degenerate encoding set, before the per-voxel fit (dtd_covariance_pipe's first step).
% We build the xps + merge here (rather than consuming a pre-combined series), so check it.
dtd_covariance_check_xps(s.xps, opt);

% --- step2: elastix inter-series motion/eddy co-registration (chrol step2_motion_correction_kth) ---
% LTE and STE are eddy/topup-corrected per series upstream but NOT co-aligned to each other; the
% covariance fit fuses them per voxel, so residual inter-series motion inflates MD / suppresses FA.
% Reference = b<=1100 s/mm^2; affine elastix; mec_b0 then extrapolation-based mec_eb. Needs elastix on PATH.
opt_mec = mdm_opt(); opt_mec.do_overwrite = 1;
s_ref = mdm_s_subsample(s, s.xps.b <= 1.1e9, op, opt_mec);
p_fn  = elastix_p_write(elastix_p_affine(100), fullfile(op, 'p_affine.txt'));
s_ref = mdm_mec_b0(s_ref, p_fn, op, opt_mec);
s     = mdm_mec_eb(s, s_ref, p_fn, op, opt_mec);            % -> LTE_STE_mc.nii.gz (motion-corrected)
fprintf('step2 elastix motion-correction done -> %s\n', s.nii_fn);

% --- Christoffer PD-paper covariance fit, steps replicated explicitly to keep our output names ---
s = mdm_s_smooth(s, opt.filter_sigma, op, opt);             % smoothing (filter_sigma=0.7)
s = mdm_s_mask(s, @mio_mask_threshold, op, opt);            % auto-mask (same as dtd_covariance_pipe)
% Stable grid reference for prepare_dmri_tensor (02), so 02 does not depend on md-dmri's internal
% intermediate naming (the mask filename encodes the _mc/_s/mask suffixes).
copyfile(s.mask_fn, fullfile(op, 'dmri_grid_ref.nii.gz'));
mfs_fn = dtd_covariance_4d_data2fit(s, fullfile(op, 'cov_mfs.mat'), opt);
dtd_covariance_4d_fit2param(mfs_fn, fullfile(op, 'cov_dps.mat'), opt);

disp('Cohort QTI covariance fit complete: qti_cov/cov_mfs.mat + cov_dps.mat');
