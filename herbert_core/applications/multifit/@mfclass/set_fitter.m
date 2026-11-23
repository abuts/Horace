function obj = set_fitter(obj, val)
% Set the optimisation algorithm used by fit()
%
% Reset to the configured default (hor_config().fitting_method, default 'lsqr'):
%   >> obj = obj.set_fitter
%   >> obj = obj.set_fitter('-default')
%
% Select one of the built-in fitters by name:
%   >> obj = obj.set_fitter('lsqr')          % Levenberg-Marquardt
%   >> obj = obj.set_fitter('fminsearch')    % Nelder-Mead simplex
%   >> obj = obj.set_fitter('lsqnonlin')     % Bounded trust-region LSQR
%   >> obj = obj.set_fitter('particleswarm') % Global particle swarm
%
% Set a custom fitter subclassing mf_fitter_base:
%   >> obj = obj.set_fitter(fitter_obj)
%
% Input:
% ------
%   val     Scalar mf_fitter_base object, fitter name ('lsqr', 'fminsearch',
%           'lsqnonlin', 'particleswarm'), or '-default'.

if nargin < 2 || isempty(val)
    obj.fitter_is_explicit_ = false;
    obj.fitter_ = mf_get_fitter();
    return
end

if (ischar(val) || (isstring(val) && isscalar(val)))
    key = lower(char(val));
    if strcmp(key, '-default') || strcmp(key, 'default') || ...
            (numel(key) >= 2 && strncmp(key, '-default', numel(key)))
        obj.fitter_is_explicit_ = false;
        obj.fitter_ = mf_get_fitter();
        return
    end
end

try
    resolved = mf_get_fitter(val);
catch ME
    if strcmp(ME.identifier, 'HERBERT:mf_get_fitter:invalid_argument')
        error('HERBERT:mfclass:invalid_argument', '%s', ME.message);
    else
        rethrow(ME);
    end
end

obj.fitter_ = resolved;
obj.fitter_is_explicit_ = true;
end
