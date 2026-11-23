classdef (Abstract) mf_fitter_base
% Abstract base class for optimization algorithms used by mfclass
%
% Subclasses implement a replaceable fitting algorithm strategy that
% optimizes free parameters for multi-dataset foreground and background
% models.
%
% Implementing a custom fitter:
%   Subclass mf_fitter_base, define the constant `name` property, and
%   either:
%     1) Override minimize_(obj, eval_fn, yval, wt, pfin, p_info, options)
%        to optimize a vector residual problem using eval_fn(p) -> f, or
%     2) Override fit(obj, w, xye, func, bfunc, pin, bpin, ...
%        f_pass_caller_info, bf_pass_caller_info, pfin, p_info, ...
%        options, perform_fit) for full control over evaluation state and
%        parallel execution.

    properties (Abstract, Constant)
        name    % Short identifier for the optimization algorithm
    end

    properties (Access = protected)
        bounds_ = []            % [] or struct('lb', col_vec, 'ub', col_vec)
        fitter_params_ = struct() % Extra solver-specific options
    end

    properties (Dependent)
        bounds          % Parameter fitting ranges ([lb, ub] struct or [])
        fitter_params   % Struct of additional fitter-specific parameters
        supports_bounds % True if fitter enforces parameter bounds
        requires_bounds % True if fitter requires finite parameter bounds
    end

    methods
        function out = get.bounds(obj)
            out = obj.bounds_;
        end

        function obj = set.bounds(obj, val)
            obj = obj.set_bounds(val);
        end

        function out = get.fitter_params(obj)
            out = obj.fitter_params_;
        end

        function obj = set.fitter_params(obj, val)
            obj = obj.set_fitter_params(val);
        end

        function tf = get.supports_bounds(obj)
            tf = obj.supports_bounds_();
        end

        function tf = get.requires_bounds(obj)
            tf = obj.requires_bounds_();
        end

        function obj = set_bounds(obj, varargin)
            % Set or clear parameter lower/upper fitting ranges.
            %
            %   >> obj = obj.set_bounds()           % Clear bounds
            %   >> obj = obj.set_bounds([])         % Clear bounds
            %   >> obj = obj.set_bounds(lb, ub)     % Lower/upper bound vectors
            %   >> obj = obj.set_bounds(ranges_nx2) % N x 2 [min, max] matrix
            %   >> obj = obj.set_bounds(bounds_struct) % struct('lb', lb, 'ub', ub)
            if isempty(varargin) || (isscalar(varargin) && isempty(varargin{1}))
                obj.bounds_ = [];
                return
            end

            if isscalar(varargin)
                arg = varargin{1};
                if isstruct(arg) && isscalar(arg) && isfield(arg, 'lb') && isfield(arg, 'ub')
                    lb = arg.lb;
                    ub = arg.ub;
                elseif isnumeric(arg) && ismatrix(arg) && size(arg, 2) == 2
                    lb = arg(:, 1);
                    ub = arg(:, 2);
                else
                    error('HERBERT:mf_fitter_base:invalid_argument', ...
                        ['Bounds must be empty, an N x 2 numeric matrix [lb, ub], ', ...
                         'or a struct with fields ''lb'' and ''ub''']);
                end
            elseif numel(varargin) == 2
                lb = varargin{1};
                ub = varargin{2};
            else
                error('HERBERT:mf_fitter_base:invalid_argument', ...
                    'Expected zero, one, or two arguments specifying bounds');
            end

            if ~isnumeric(lb) || ~isnumeric(ub) || ~isreal(lb) || ~isreal(ub)
                error('HERBERT:mf_fitter_base:invalid_argument', ...
                    'Lower and upper bounds must be real numeric arrays');
            end
            lb = double(lb(:));
            ub = double(ub(:));
            if numel(lb) ~= numel(ub)
                error('HERBERT:mf_fitter_base:invalid_argument', ...
                    'Lower bounds length (%d) must match upper bounds length (%d)', ...
                    numel(lb), numel(ub));
            end
            if any(isnan(lb)) || any(isnan(ub))
                error('HERBERT:mf_fitter_base:invalid_argument', ...
                    'Bounds must not contain NaN values');
            end
            if any(lb > ub)
                error('HERBERT:mf_fitter_base:invalid_argument', ...
                    'Lower bounds must be <= corresponding upper bounds');
            end

            obj.bounds_ = struct('lb', lb, 'ub', ub);
        end

        function obj = set_fitter_params(obj, varargin)
            % Set additional algorithm-specific fitter parameters.
            %
            %   >> obj = obj.set_fitter_params()          % Clear extra params
            %   >> obj = obj.set_fitter_params(param_struct)
            %   >> obj = obj.set_fitter_params('key1', val1, 'key2', val2, ...)
            if isempty(varargin) || (isscalar(varargin) && isempty(varargin{1}))
                obj.fitter_params_ = struct();
                return
            end

            if isscalar(varargin)
                arg = varargin{1};
                if ischar(arg) || isstring(arg)
                    key = lower(char(arg));
                    if strcmp(key, '-default') || strcmp(key, 'default')
                        obj.fitter_params_ = struct();
                        obj.bounds_ = [];
                        return
                    end
                    error('HERBERT:mf_fitter_base:invalid_argument', ...
                        'Unrecognised fitter parameter flag "%s"', char(arg));
                elseif isstruct(arg) && isscalar(arg)
                    s_in = arg;
                else
                    error('HERBERT:mf_fitter_base:invalid_argument', ...
                        'Fitter parameters must be a scalar struct or name-value pairs');
                end
            elseif mod(numel(varargin), 2) == 0
                s_in = struct();
                for k = 1:2:numel(varargin)
                    fname = varargin{k};
                    if ~(ischar(fname) || (isstring(fname) && isscalar(fname))) || isempty(char(fname))
                        error('HERBERT:mf_fitter_base:invalid_argument', ...
                            'Fitter parameter names must be non-empty strings');
                    end
                    s_in.(char(fname)) = varargin{k+1};
                end
            else
                error('HERBERT:mf_fitter_base:invalid_argument', ...
                    'Fitter parameters must be supplied as a struct or name-value pairs');
            end

            fnames = fieldnames(s_in);
            for k = 1:numel(fnames)
                fn = fnames{k};
                if strcmpi(fn, 'bounds') || strcmpi(fn, 'fitting_ranges')
                    obj = obj.set_bounds(s_in.(fn));
                else
                    obj.fitter_params_.(fn) = s_in.(fn);
                end
            end
        end

        function [p_best, sig, cor, chisqr_red, converged] = fit(obj, w, xye, ...
                func, bfunc, pin, bpin, f_pass_caller_info, bf_pass_caller_info, ...
                pfin, p_info, options, perform_fit)
            % Perform optimization (or chi-squared evaluation) over datasets w.
            %
            % Inputs match mfclass / multifit_lsqr conventions, with `options`
            % holding the mfclass options structure (listing,
            % fit_control_parameters, etc.).

            if nargin < 13 || isempty(perform_fit)
                perform_fit = true;
            end

            [yval, wt] = obj.extract_data_and_weights_(w, xye);
            nval = numel(yval);
            npfree = numel(pfin);
            nnorm = max(nval - npfree, 1);
            if nval < npfree
                error('HERBERT:mf_fitter_base:invalid_argument', ...
                    'Number of data points (%d) must be >= number of free parameters (%d)', ...
                    nval, npfree);
            end

            listing = 0;
            if isstruct(options) && isfield(options, 'listing')
                listing = options.listing;
            end

            S = [];
            Store = [];
            function [f, S_out, Store_out] = eval_model(p, store_calc)
                if nargin < 2
                    store_calc = false;
                end
                [f_cell, ~, S_out, Store_out] = multifit_lsqr_func_eval( ...
                    w, xye, func, bfunc, pin, bpin, ...
                    f_pass_caller_info, bf_pass_caller_info, ...
                    p(:), p_info, store_calc, S, Store, listing);
                f = cat(1, f_cell{:});
                if store_calc
                    S = S_out;
                    Store = Store_out;
                end
            end

            if ~perform_fit
                p_best = pfin(:);
                sig = zeros(npfree, 1);
                cor = zeros(npfree);
                f = eval_model(pfin, false);
                resid = wt .* (yval - f);
                chisqr_red = (resid' * resid) / nnorm;
                converged = false;
                return
            end

            % Seed internal state (e.g. Monte Carlo RNG states) and lookup
            % caches once before iterative optimization so subsequent
            % evaluations reset RNG state reproducibly.
            eval_model(pfin, true);

            [p_best, sig, cor, chisqr_red, converged] = obj.minimize_( ...
                @eval_model, yval, wt, pfin(:), p_info, options);
            p_best = p_best(:);
            sig = sig(:);
        end
    end

    methods (Access = protected)
        function tf = supports_bounds_(~)
            tf = false;
        end

        function tf = requires_bounds_(~)
            tf = false;
        end

        function [p_best, sig, cor, chisqr_red, converged] = minimize_(obj, ~, ~, ~, ~, ~, ~) %#ok<STOUT>
            % Optimize free parameters given vector model evaluator eval_fn(p).
            %
            % Subclasses that do not override fit() must override this method.
            error('HERBERT:mf_fitter_base:not_implemented', ...
                'Fitter "%s" must implement minimize_() or override fit()', ...
                class(obj));
        end

        function [lb, ub] = resolve_bounds_(obj, npfree)
            % Resolve parameter bounds for npfree parameters and validate
            % against fitter capabilities (supports_bounds / requires_bounds).
            if isempty(obj.bounds_)
                if obj.requires_bounds
                    error('HERBERT:mf_fitter_base:missing_bounds', ...
                        ['Fitter "%s" requires finite parameter fitting ranges (bounds). ', ...
                         'Specify bounds with set_fitting_ranges(lb, ub).'], ...
                        obj.name);
                end
                lb = -inf(npfree, 1);
                ub = inf(npfree, 1);
                return
            end

            lb = obj.bounds_.lb(:);
            ub = obj.bounds_.ub(:);
            if numel(lb) ~= npfree
                error('HERBERT:mf_fitter_base:invalid_argument', ...
                    ['Number of bound elements (%d) does not match number of free ', ...
                     'parameters (%d)'], numel(lb), npfree);
            end

            if obj.requires_bounds && (any(~isfinite(lb)) || any(~isfinite(ub)))
                error('HERBERT:mf_fitter_base:missing_bounds', ...
                    ['Fitter "%s" requires finite lower and upper fitting ranges for all ', ...
                     '%d free parameters.'], obj.name, npfree);
            end

            if ~obj.supports_bounds && (any(isfinite(lb)) || any(isfinite(ub)))
                warning('HERBERT:mf_fitter_base:bounds_not_supported', ...
                    ['Fitter "%s" does not support parameter bounds; supplied fitting ranges ', ...
                     'will be ignored. Use ''lsqnonlin'', ''fminsearch'', or ''particleswarm'' ', ...
                     'for bounded fitting.'], obj.name);
            end
        end

        function ctrl = parse_control_options_(obj, options)
            % Extract fit control parameters, listing level, and overrides from
            % fitter_params_.
            fcp = [1e-4, 20, 1e-3];
            listing = 0;
            if isstruct(options)
                if isfield(options, 'fit_control_parameters') && ~isempty(options.fit_control_parameters)
                    fcp = options.fit_control_parameters;
                end
                if isfield(options, 'listing') && ~isempty(options.listing)
                    listing = options.listing;
                end
            end

            ctrl.dp = fcp(1);
            ctrl.max_iter = fcp(2);
            ctrl.tol = fcp(3);
            ctrl.listing = listing;
            ctrl.extra = obj.fitter_params_;
        end
    end

    methods (Static, Access = protected)
        function [yval, wt] = extract_data_and_weights_(w, xye)
            % Extract column vectors of data values yval and weights 1/sigma.
            yval = cell(size(w));
            wt = cell(size(w));
            for i = 1:numel(w)
                if xye(i)
                    yval{i} = w{i}.y;
                    wt{i} = 1 ./ w{i}.e;
                else
                    [yval{i}, var_i, msk] = sigvar_get(w{i});
                    yval{i} = yval{i}(msk);
                    wt{i} = 1 ./ sqrt(var_i(msk));
                end
                yval{i} = yval{i}(:);
                wt{i} = wt{i}(:);
            end
            yval = cell2mat(yval(:));
            wt = cell2mat(wt(:));
        end

        function [sig, cor] = estimate_covariance_(eval_fn, wt, p_best, chisqr_red, options)
            % Estimate parameter standard deviations and correlation matrix at
            % p_best using finite-difference Jacobian identical to multifit_lsqr.
            npfree = numel(p_best);
            if npfree == 0
                sig = zeros(0, 1);
                cor = zeros(0, 0);
                return
            end

            dp = 1e-4;
            if isstruct(options) && isfield(options, 'fit_control_parameters') && ...
                    ~isempty(options.fit_control_parameters)
                dp = options.fit_control_parameters(1);
            end
            if dp == 0
                dp = 1e-4;
            end

            f_best = eval_fn(p_best, false);
            jac = zeros(numel(f_best), npfree);
            min_abs_del = 1e-12;

            for j = 1:npfree
                del = dp * p_best(j);
                if abs(del) <= min_abs_del
                    if p_best(j) >= 0
                        del = min_abs_del;
                    else
                        del = -min_abs_del;
                    end
                end
                if dp >= 0
                    ppos = p_best;
                    ppos(j) = p_best(j) + del;
                    plus = eval_fn(ppos, false);
                    jac(:, j) = (plus - f_best) / del;
                else
                    ppos = p_best;
                    ppos(j) = p_best(j) + del;
                    pneg = p_best;
                    pneg(j) = p_best(j) - del;
                    plus = eval_fn(ppos, false);
                    minus = eval_fn(pneg, false);
                    jac(:, j) = (plus - minus) / (2 * del);
                end
            end

            [sig, cor] = mf_fitter_base.covariance_from_jacobian_(jac, wt, chisqr_red);
        end

        function [sig, cor] = covariance_from_jacobian_(jac, wt, chisqr_red)
            % Compute parameter errors sig and correlation matrix cor from a
            % Jacobian matrix using SVD with singular-value conditioning.
            npfree = size(jac, 2);
            if npfree == 0
                sig = zeros(0, 1);
                cor = zeros(0, 0);
                return
            end

            if ~isempty(wt)
                for k = 1:npfree
                    jac(:, k) = wt(:) .* jac(:, k);
                end
            end

            [~, s, v] = svd(jac, 0);
            max_sv = max(abs(s(:)));
            if max_sv == 0
                sig = zeros(npfree, 1);
                cor = zeros(npfree);
                return
            end

            degraded = abs(s / max_sv) < eps('single');
            if any(diag(degraded))
                nrm = zeros(npfree, 1);
                for k = 1:npfree
                    nrm(k) = jac(:, k)' * jac(:, k);
                    if nrm(k) > 0
                        nrm(k) = 1 / sqrt(nrm(k));
                    end
                    jac(:, k) = nrm(k) * jac(:, k);
                end
                [~, s, v] = svd(jac, 0);
                max_sv = max(abs(s(:)));
                if max_sv == 0
                    sig = zeros(npfree, 1);
                    cor = zeros(npfree);
                    return
                end
                degraded = abs(s / max_sv) < eps('single');
            else
                nrm = ones(npfree, 1);
            end

            s(degraded) = 0;
            diag_s = diag(s);
            inv_s = zeros(size(diag_s));
            nonzero = abs(diag_s) > 0;
            inv_s(nonzero) = 1 ./ diag_s(nonzero);
            v = v .* repmat(inv_s', [npfree, 1]);

            cov = chisqr_red * (v * v');
            dcov = diag(cov);
            sig = sqrt(max(dcov, 0)) .* nrm;

            inv_sqrt_cov = zeros(size(dcov));
            pos = dcov > 0;
            inv_sqrt_cov(pos) = 1 ./ sqrt(dcov(pos));
            cor = (inv_sqrt_cov * inv_sqrt_cov') .* cov;
        end
    end
end
