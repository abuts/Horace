function [cov_proj, cov_spec, cov_hkle] = tobyfit_DGdisk_resfun_covariance (win, ipix)
% Deprecated thin wrapper delegating to tobyfit_DGdisk_res_model.covariance
%
% See also tobyfit_DGdisk_res_model

res_model = tobyfit_DGdisk_res_model();
[cov_proj, cov_spec, cov_hkle] = res_model.covariance(win, ipix);
end
