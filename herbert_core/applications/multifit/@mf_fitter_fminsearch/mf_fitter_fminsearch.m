classdef mf_fitter_fminsearch < mf_fitter_base
% Nelder-Mead simplex derivative-free optimizer using MATLAB fminsearch
%
% Suitable for noisy or non-smooth objective landscapes (such as Monte Carlo
% resolution convolution with small mc_points) where finite-difference
% Jacobians in Levenberg-Marquardt may struggle. Supports optional parameter
% fitting ranges (bounds) without requiring any additional MATLAB toolboxes.

    properties (Constant)
        name = 'fminsearch'
    end

    methods
        function obj = mf_fitter_fminsearch()
            % Construct Nelder-Mead simplex fitter.
        end
    end

    methods (Access = protected)
        function tf = supports_bounds_(~)
            tf = true;
        end

        function [p_best, sig, cor, chisqr_red, converged] = minimize_( ...
                obj, eval_fn, yval, wt, pfin, ~, options)
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
                    warning('HERBERT:mf_fitter_fminsearch:initial_out_of_bounds', ...
                        'Initial parameter(s) clamped to supplied fitting ranges');
                    p0 = min(max(p0, lb), ub);
                end
            end

            ctrl = obj.parse_control_options_(options);
            max_iter = max(200 * npfree, ctrl.max_iter);
            if isfield(ctrl.extra, 'MaxIter')
                max_iter = ctrl.extra.MaxIter;
            elseif isfield(ctrl.extra, 'MaxIterations')
                max_iter = ctrl.extra.MaxIterations;
            end
            max_evals = max(200 * npfree, max_iter * 2);
            if isfield(ctrl.extra, 'MaxFunEvals')
                max_evals = ctrl.extra.MaxFunEvals;
            elseif isfield(ctrl.extra, 'MaxFunctionEvaluations')
                max_evals = ctrl.extra.MaxFunctionEvaluations;
            end
            tol_x = min(1e-6, abs(ctrl.tol));
            if isfield(ctrl.extra, 'TolX')
                tol_x = ctrl.extra.TolX;
            end
            tol_fun = min(1e-6, abs(ctrl.tol));
            if isfield(ctrl.extra, 'TolFun')
                tol_fun = ctrl.extra.TolFun;
            end

            if ctrl.listing > 0
                disp_mode = 'iter';
                disp('----------------------------------------------------')
                fprintf(['Beginning fminsearch fit (Nelder-Mead simplex) with: \n' ...
                    'max %d iterations; max %d func eval;\n' ...
                    'tol_x %g;  TolFun %g\n'], ...
                    max_iter,max_evals,tol_x,tol_fun);
                disp('----------------------------------------------------')
            else
                disp_mode = 'off';
            end

            opts = optimset('Display', disp_mode, ...
                'MaxIter', max_iter, ...
                'MaxFunEvals', max_evals, ...
                'TolX', tol_x, ...
                'TolFun', tol_fun);

            function chi2 = obj_fun(p)
                if has_bounds && any(p(:) < lb | p(:) > ub)
                    chi2 = inf;
                    return
                end
                f = eval_fn(p(:), false);
                r = wt .* (yval - f);
                chi2 = r' * r;
            end

            [p_best, fval, exitflag] = fminsearch(@obj_fun, p0, opts);
            p_best = p_best(:);
            chisqr_red = fval / nnorm;
            converged = (exitflag == 1);
            if ~converged
                warning('HERBERT:mf_fitter_fminsearch:not_converged', ...
                    'WARNING: fminsearch did not converge within iteration/evaluation limit');
            end

            [sig, cor] = obj.estimate_covariance_(eval_fn, wt, p_best, chisqr_red, options);
        end
    end
end
