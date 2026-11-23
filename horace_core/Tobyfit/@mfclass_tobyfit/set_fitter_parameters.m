function obj = set_fitter_parameters(obj, varargin)
% Set additional optimizer parameters or fitting ranges for Tobyfit
%
% Clear extra fitter parameters (or reset both params and ranges):
%   >> obj = obj.set_fitter_parameters
%   >> obj = obj.set_fitter_parameters([])
%   >> obj = obj.set_fitter_parameters('-default')
%
% Specify parameters via name-value pairs or a structure:
%   >> obj = obj.set_fitter_parameters('fitting_ranges', ranges)
%   >> obj = obj.set_fitter_parameters('SwarmSize', 30, 'MaxIterations', 200)
%   >> obj = obj.set_fitter_parameters(struct('MaxIterations', 200, 'FunctionTolerance', 1e-5))
%
% Any 'fitting_ranges' or 'bounds' key updates obj.fitting_ranges. All other
% fields are forwarded to the active mf_fitter_base optimizer.

try
    obj = set_fitter_parameters@mfclass(obj, varargin{:});
catch ME
    if strcmp(ME.identifier, 'HERBERT:mfclass:invalid_argument')
        error('HORACE:mfclass_tobyfit:invalid_argument', '%s', ME.message);
    else
        rethrow(ME);
    end
end
end
