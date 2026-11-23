classdef mf_fitter_lsqnonlin < mf_fitter_base
% Bounded Trust-Region-Reflective / Levenberg-Marquardt nonlinear least-squares
% fitter backed by MATLAB Optimization Toolbox `lsqnonlin`.
%
% Supports optional box constraints (fitting ranges [lb, ub]) natively and
% computes parameter uncertainties and correlation matrix from the returned
% weighted Jacobian.

    properties (Constant)
        name = 'lsqnonlin'
    end

    methods
        function obj = mf_fitter_lsqnonlin()
            % Construct lsqnonlin least-squares fitter.
        end
    end

    methods (Static)
        function tf = is_available()
            % Return true when Optimization Toolbox `lsqnonlin` is available.
            tf = ~isempty(which('lsqnonlin'));
        end
    end

    methods (Access = protected)
        function tf = supports_bounds_(~)
            tf = true;
        end

        function [p_best, sig, cor, chisqr_red, converged] = minimize_( ...
                obj, eval_fn, yval, wt, pfin, ~, options)
            if ~obj.is_available()
                error('HERBERT:mf_fitter_lsqnonlin:not_available', ...
                    'Fitter ''lsqnonlin'' requires Optimization Toolbox (lsqnonlin not found)');
            end

            npfree = numel(pfin);
            nnorm = max(numel(yval) - npfree, 1);

            if npfree == 0
                f0 = eval_fn(pfin, false);
                r0 = wt .* (yval - f0);
                p_best = pfin(:);
                sig = zeros(0, 1);
                cor = zeros(0, 0);
                chisqr_red = (r0' * r0) / nnorm;
                converged = true;
                return
            end

            [lb, ub] = obj.resolve_bounds_(npfree);
            has_bounds = any(isfinite(lb)) || any(isfinite(ub));
            p0 = pfin(:);
            if has_bounds
                out_of_bounds = (p0 < lb) | (p0 > ub);
                if any(out_of_bounds)
                    warning('HERBERT:mf_fitter_lsqnonlin:initial_out_of_bounds', ...
                        'Initial parameter(s) clamped to supplied fitting ranges');
                    p0 = min(max(p0, lb), ub);
                end
                lb_arg = lb;
                ub_arg = ub;
                alg = 'trust-region-reflective';
            else
                lb_arg = [];
                ub_arg = [];
                alg = 'trust-region-reflective';
            end

            ctrl = obj.parse_control_options_(options);
            if isfield(ctrl.extra, 'Algorithm')
                alg = ctrl.extra.Algorithm;
            end
            if ctrl.dp < 0
                fd_type = 'central';
            else
                fd_type = 'forward';
            end
            fd_step = max(abs(ctrl.dp), 1e-12);

            max_iter = max(100, ctrl.max_iter);
            if isfield(ctrl.extra, 'MaxIterations')
                max_iter = ctrl.extra.MaxIterations;
            elseif isfield(ctrl.extra, 'MaxIter')
                max_iter = ctrl.extra.MaxIter;
            end
            tol_fun = min(1e-6, max(abs(ctrl.tol), 1e-12));
            if isfield(ctrl.extra, 'FunctionTolerance')
                tol_fun = ctrl.extra.FunctionTolerance;
            end
            tol_step = tol_fun;
            if isfield(ctrl.extra, 'StepTolerance')
                tol_step = ctrl.extra.StepTolerance;
            end

            if ctrl.listing > 0
                disp_mode = 'iter';
            else
                disp_mode = 'off';
            end

            opts = optimoptions('lsqnonlin', ...
                'Algorithm', alg, ...
                'Display', disp_mode, ...
                'FiniteDifferenceStepSize', fd_step, ...
                'FiniteDifferenceType', fd_type, ...
                'MaxIterations', max_iter, ...
                'FunctionTolerance', tol_fun, ...
                'StepTolerance', tol_step);
            if isfield(ctrl.extra, 'MaxFunctionEvaluations')
                opts.MaxFunctionEvaluations = ctrl.extra.MaxFunctionEvaluations;
            end

            function r = resid_fun(p)
                f = eval_fn(p(:), false);
                r = wt .* (f - yval);
            end

            [p_best, resnorm, ~, exitflag, ~, ~, jacobian] = ...
                lsqnonlin(@resid_fun, p0, lb_arg, ub_arg, opts);
            p_best = p_best(:);
            chisqr_red = resnorm / nnorm;
            converged = (exitflag > 0);
            if ~converged
                warning('HERBERT:mf_fitter_lsqnonlin:not_converged', ...
                    'WARNING: lsqnonlin did not converge (exitflag = %d)', exitflag);
            end

            [sig, cor] = obj.covariance_from_jacobian_(full(jacobian), [], chisqr_red);
        end
    end
end
