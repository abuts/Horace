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

    methods
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
                    store_calc = true;
                end
                [f_cell, ~, S_out, Store_out] = multifit_lsqr_func_eval( ...
                    w, xye, func, bfunc, pin, bpin, ...
                    f_pass_caller_info, bf_pass_caller_info, ...
                    p, p_info, store_calc, S, Store, listing);
                f = cat(1, f_cell{:});
                if store_calc
                    S = S_out;
                    Store = Store_out;
                end
            end

            if ~perform_fit
                p_best = pfin;
                sig = zeros(1, npfree);
                cor = zeros(npfree);
                f = eval_model(pfin, false);
                resid = wt .* (yval - f);
                chisqr_red = (resid' * resid) / nnorm;
                converged = false;
                return
            end

            [p_best, sig, cor, chisqr_red, converged] = obj.minimize_( ...
                @eval_model, yval, wt, pfin, p_info, options);
        end
    end

    methods (Access = protected)
        function [p_best, sig, cor, chisqr_red, converged] = minimize_(obj, ~, ~, ~, ~, ~, ~) %#ok<STOUT>
            % Optimize free parameters given vector model evaluator eval_fn(p).
            %
            % Subclasses that do not override fit() must override this method.
            error('HERBERT:mf_fitter_base:not_implemented', ...
                'Fitter "%s" must implement minimize_() or override fit()', ...
                class(obj));
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
    end
end
