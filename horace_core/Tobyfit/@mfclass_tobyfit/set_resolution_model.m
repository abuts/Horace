function obj = set_resolution_model(obj, res_model)
% Set the resolution convolution model for Tobyfit
%
%   >> obj = obj.set_resolution_model(res_model)
%
% Input:
% ------
%   res_model   Scalar instance of a tobyfit_res_model_base subclass, or an
%               instrument object / name supported by tobyfit_get_res_model
%               (e.g. 'IX_inst_DGfermi', 'IX_inst_DGdisk').
%
% Extensibility:
%   Additional resolution models (for example, analytical Gaussian covariance
%   models) can be enabled later by subclassing tobyfit_res_model_base and
%   passing an instance or registering its identifier in tobyfit_get_res_model.

if nargin < 2 || isempty(res_model)
    error('HORACE:mfclass_tobyfit:invalid_argument', ...
        'A resolution model instance or instrument identifier must be provided');
end

res_model = tobyfit_get_res_model(res_model);
obj.res_model_ = res_model;

% Bind wrapper function and initialisation function to the active model
obj.wrapfun = mfclass_wrapfun(@tobyfit_resconv, [], @func_eval, [], ...
    true, false, @(win) res_model.init(win), []);

% Initialise or reconcile MC contribution options for the new model
valid_contr = res_model.mc_contribution_names();
if isempty(obj.mc_contributions_) || ...
        ~isequal(valid_contr(:), fieldnames(obj.mc_contributions_))
    obj = obj.set_mc_contributions();
end
end
