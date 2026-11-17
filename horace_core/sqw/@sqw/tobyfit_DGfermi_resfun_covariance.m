function [cov_proj, cov_spec, cov_hkle] = tobyfit_DGfermi_resfun_covariance (win, ipix)
% Deprecated thin wrapper delegating to tobyfit_DGfermi_res_model.covariance
%
% See also tobyfit_DGfermi_res_model

res_model = tobyfit_DGfermi_res_model();
[cov_proj, cov_spec, cov_hkle] = res_model.covariance(win, ipix);
end
