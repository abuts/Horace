function obj = set_fitter(obj, val)
% Set the optimisation algorithm used by fit()
%
% Set default Levenberg-Marquardt least-squares fitter:
%   >> obj = obj.set_fitter
%   >> obj = obj.set_fitter('-default')
%   >> obj = obj.set_fitter('lsqr')
%
% Set a custom fitter subclassing mf_fitter_base:
%   >> obj = obj.set_fitter(fitter_obj)
%
% Future built-in optimizers (e.g. Nelder-Mead / fminsearch or lsqnonlin)
% can be enabled by subclassing mf_fitter_base and registering their key
% name here.
%
% Input:
% ------
%   val     Scalar mf_fitter_base object, or '-default' / 'lsqr'.

if nargin < 2 || isempty(val)
    obj.fitter_ = mf_fitter_lsqr();
    return
end

if isa(val, 'mf_fitter_base')
    if ~isscalar(val)
        error('HERBERT:mfclass:invalid_argument', ...
            'Fitter object must be a scalar mf_fitter_base instance');
    end
    obj.fitter_ = val;
elseif ischar(val) || isstring(val)
    key = lower(char(val));
    if strcmp(key, '-default') || strcmp(key, 'lsqr') || ...
            (numel(key) >= 2 && strncmp(key, '-default', numel(key)))
        obj.fitter_ = mf_fitter_lsqr();
    else
        error('HERBERT:mfclass:invalid_argument', ...
            'Unrecognised fitter name "%s"', char(val));
    end
else
    error('HERBERT:mfclass:invalid_argument', ...
        'Fitter must be an mf_fitter_base instance or ''lsqr''/''-default''');
end
end
