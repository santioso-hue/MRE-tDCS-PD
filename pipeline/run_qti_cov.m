% run_qti_cov.m — QTI covariance fit (md-dmri toolbox, Westin et al. 2016) -> oriented mean tensor <D>.
%
% Upstream step of the MD-dMRI pipeline: produces qti_cov/cov_mfs.mat (model fit; the oriented mean
% tensor is m(:,:,:,2:7)) and cov_dps.mat (MD, uFA, principal eigenvector u). prepare_dmri_tensor.py
% reads these. Run this once in MATLAB before pipeline/02_register_dmri_to_T1.sh.
%
% The constrained/regularized/heteroscedastic covariance fit is the standard QTI estimate of the
% macroscopic mean diffusion tensor — not magnitude-inflated and with a reliable eigenframe, unlike
% the full DTD Monte-Carlo mean. Fit runs on the already motion/eddy-corrected (+smoothed) series, so
% no extra smoothing here (we call data2fit directly rather than dtd_covariance_pipe).
%
% Usage (paths come from config/config.sh via environment):
%   FIT_DIR=/path/to/fit MDDMRI_DIR=/path/to/md-dmri matlab -batch "run('pipeline/run_qti_cov.m')"

fit_dir = getenv('FIT_DIR');
if isempty(fit_dir), error('Set FIT_DIR (see config/config.sh), e.g. FIT_DIR=$FIT_DIR'); end
mddmri = getenv('MDDMRI_DIR');
if isempty(mddmri), error('Set MDDMRI_DIR to the md-dmri toolbox root (containing setup_paths.m)'); end

run(fullfile(mddmri, 'setup_paths.m'));

op = fullfile(fit_dir, 'qti_cov');
if ~exist(op, 'dir'), mkdir(op); end

% motion/eddy-corrected (+smoothed) series, the same input the upstream fit used; auto-loads *_xps.mat
s = mdm_s_from_nii(fullfile(fit_dir, 'LTE_STE_PTE_mc_s.nii.gz'));
s.mask_fn = fullfile(fit_dir, 'LTE_STE_PTE_mc_s_mask.nii.gz');

opt = dtd_covariance_opt(mdm_opt());
opt.do_overwrite = 1;

mfs_fn = dtd_covariance_4d_data2fit(s, fullfile(op, 'cov_mfs.mat'), opt);
dtd_covariance_4d_fit2param(mfs_fn, fullfile(op, 'cov_dps.mat'), opt);

disp('QTI covariance fit complete: qti_cov/cov_mfs.mat + cov_dps.mat');
