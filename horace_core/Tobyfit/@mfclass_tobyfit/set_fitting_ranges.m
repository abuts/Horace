function obj = set_fitting_ranges(obj, varargin)
% Set or clear parameter fitting ranges [lb, ub] for bounded Tobyfit optimizers
%
% Clear fitting ranges (unconstrained):
%   >> obj = obj.set_fitting_ranges
%   >> obj = obj.set_fitting_ranges([])
%   >> obj = obj.set_fitting_ranges('-default')
%
% Set lower and upper bounds for fit parameters:
%   >> obj = obj.set_fitting_ranges(lb, ub)
%   >> obj = obj.set_fitting_ranges([min_1, max_1; min_2, max_2; ...])
%   >> obj = obj.set_fitting_ranges(struct('lb', lb, 'ub', ub))
%
% Notes:
%   - Bounds are validated and mapped onto the free parameters when fit()
%     is executed. They may be supplied for all foreground/background
%     parameters or directly for the free parameters.
%   - Required by global/bounded optimizers such as 'particleswarm' (set via
%     hor_config().fitting_method or set_fitter) and supported by
%     'lsqnonlin' and 'fminsearch'.
%
% Example:
%   >> set(hor_config, 'fitting_method', 'particleswarm')
%   >> kk = tobyfit(w);
%   >> kk = kk.set_fun(@sqw_eval_fn, [100, 50, 20]);
%   >> kk = kk.set_fitting_ranges([0, 500; 10, 100; 1, 50]);
%   >> [wfit, fp] = kk.fit();

try
    obj = set_fitting_ranges@mfclass(obj, varargin{:});
catch ME
    if strcmp(ME.identifier, 'HERBERT:mfclass:invalid_argument')
        error('HORACE:mfclass_tobyfit:invalid_argument', '%s', ME.message);
    else
        rethrow(ME);
    end
end
end
