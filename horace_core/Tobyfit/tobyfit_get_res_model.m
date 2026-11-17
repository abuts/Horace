function res_model = tobyfit_get_res_model(inst)
% Resolve default Tobyfit resolution model for a given instrument or class name
%
%   >> res_model = tobyfit_get_res_model(inst)
%
% Input:
% ------
%   inst        Instrument object (IX_inst subclass), instrument class name
%               (e.g. 'IX_inst_DGfermi', 'IX_inst_DGdisk'), or a
%               tobyfit_res_model_base instance.
%
% Output:
% -------
%   res_model   Instance of a tobyfit_res_model_base subclass.
%
% Extensibility:
%   Additional resolution models (e.g. analytical Gaussian covariance models)
%   can be enabled in future by subclassing tobyfit_res_model_base and adding
%   a case here or passing an instance via tobyfit(..., 'res_model', model).

if isa(inst, 'tobyfit_res_model_base') && isscalar(inst)
    res_model = inst;
    return
end

if ischar(inst) || isstring(inst)
    inst_name = char(inst);
elseif isa(inst, 'IX_inst')
    inst_name = class(inst);
else
    error('HORACE:tobyfit_get_res_model:invalid_argument', ...
        'Instrument must be an IX_inst object or class name, got %s', class(inst));
end

switch inst_name
    case {'IX_inst_DGfermi', 'DGfermi'}
        res_model = tobyfit_DGfermi_res_model();
    case {'IX_inst_DGdisk', 'DGdisk'}
        res_model = tobyfit_DGdisk_res_model();
    otherwise
        error('HORACE:tobyfit_get_res_model:invalid_argument', ...
            'No resolution function model implemented for instrument "%s"', inst_name);
end
end
