function obj = set_fitting_ranges(obj, varargin)
% Set or clear parameter fitting ranges (lower and upper bounds)
%
% Clear fitting ranges (unconstrained):
%   >> obj = obj.set_fitting_ranges
%   >> obj = obj.set_fitting_ranges([])
%   >> obj = obj.set_fitting_ranges('-default')
%
% Set lower and upper bounds:
%   >> obj = obj.set_fitting_ranges(lb, ub)
%   >> obj = obj.set_fitting_ranges([min_1, max_1; min_2, max_2; ...])
%   >> obj = obj.set_fitting_ranges(struct('lb', lb, 'ub', ub))
%
% Input:
% ------
%   lb, ub      Real numeric vectors (same length as free parameters or full
%               foreground/background parameter list), with lb <= ub.
%               Use -Inf or +Inf for half-unbounded parameters when supported
%               by the active optimizer (note: 'particleswarm' requires finite
%               ranges on all free parameters).

if numel(varargin) == 0 || (numel(varargin) == 1 && isempty(varargin{1}))
    obj.fitting_ranges_ = [];
    if obj.fitter_is_explicit_
        obj.fitter_ = obj.fitter_.set_bounds([]);
    end
    return
end

if numel(varargin) == 1 && (ischar(varargin{1}) || (isstring(varargin{1}) && isscalar(varargin{1})))
    key = lower(char(varargin{1}));
    if strcmp(key, '-default') || strcmp(key, 'default') || ...
            (numel(key) >= 2 && strncmp(key, '-default', numel(key)))
        obj.fitting_ranges_ = [];
        if obj.fitter_is_explicit_
            obj.fitter_ = obj.fitter_.set_bounds([]);
        end
        return
    end
end

try
    tmp = obj.fitter_.set_bounds(varargin{:});
catch ME
    if strcmp(ME.identifier, 'HERBERT:mf_fitter_base:invalid_argument')
        error('HERBERT:mfclass:invalid_argument', '%s', ME.message);
    else
        rethrow(ME);
    end
end

obj.fitting_ranges_ = tmp.bounds;
if obj.fitter_is_explicit_
    obj.fitter_ = tmp;
end
end
