function [ok, mess, lookup, npix] = tobyfit_DGfermi_resconv_init (varargin)
% Deprecated thin wrapper delegating to tobyfit_DGfermi_res_model.init
%
% See also tobyfit_DGfermi_res_model

res_model = tobyfit_DGfermi_res_model();
[ok, mess, lookup, npix] = res_model.init(varargin{:});
end
