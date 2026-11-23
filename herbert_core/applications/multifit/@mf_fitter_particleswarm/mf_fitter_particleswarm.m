classdef mf_fitter_particleswarm < mf_fitter_base
% Global derivative-free particle swarm optimizer backed by Global Optimization
% Toolbox `particleswarm`.
%
% This optimizer requires finite parameter fitting ranges (lower and upper
% bounds) specified via set_fitting_ranges(lb, ub) or
% set_fitter_parameters('bounds', ...) on the fitting object.

    properties (Constant)
        name = 'particleswarm'
    end

    methods
        function obj = mf_fitter_particleswarm()
            % Construct Particle Swarm global fitter.
        end
    end

    methods (Static)
        function tf = is_available()
            % Return true when Global Optimization Toolbox `particleswarm` is available.
            tf = ~isempty(which('particleswarm'));
        end
    end

    methods (Access = protected)
        function tf = supports_bounds_(~)
            tf = true;
        end

        function tf = requires_bounds_(~)
            tf = true;
        end

        function [p_best, sig, cor, chisqr_red, converged] = minimize_( ...
                obj, eval_fn, yval, wt, pfin, ~, options)
            if ~obj.is_available()
                error('HERBERT:mf_fitter_particleswarm:not_available', ...
                    'Fitter ''particleswarm'' requires Global Optimization Toolbox (particleswarm not found)');
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
            ctrl = obj.parse_control_options_(options);

            swarm_size = min(100, max(10 * npfree, 20));
            if isfield(ctrl.extra, 'SwarmSize')
                swarm_size = ctrl.extra.SwarmSize;
            end
            max_iter = max(200, ctrl.max_iter);
            if isfield(ctrl.extra, 'MaxIterations')
                max_iter = ctrl.extra.MaxIterations;
            elseif isfield(ctrl.extra, 'MaxIter')
                max_iter = ctrl.extra.MaxIter;
            end
            tol_fun = min(1e-6, max(abs(ctrl.tol), 1e-12));
            if isfield(ctrl.extra, 'FunctionTolerance')
                tol_fun = ctrl.extra.FunctionTolerance;
            end

            if ctrl.listing > 0
                disp_mode = 'iter';
            else
                disp_mode = 'off';
            end

            opts = optimoptions('particleswarm', ...
                'Display', disp_mode, ...
                'SwarmSize', swarm_size, ...
                'MaxIterations', max_iter, ...
                'FunctionTolerance', tol_fun);

            % Seed initial parameter guess inside the swarm if it lies within bounds
            p0 = pfin(:)';
            if all(p0 >= lb' & p0 <= ub') && ~isfield(ctrl.extra, 'InitialSwarmMatrix')
                opts.InitialSwarmMatrix = p0;
            elseif isfield(ctrl.extra, 'InitialSwarmMatrix')
                opts.InitialSwarmMatrix = ctrl.extra.InitialSwarmMatrix;
            end

            if isfield(ctrl.extra, 'MaxStallIterations')
                opts.MaxStallIterations = ctrl.extra.MaxStallIterations;
            end
            if isfield(ctrl.extra, 'HybridFcn')
                opts.HybridFcn = ctrl.extra.HybridFcn;
            end
            if isfield(ctrl.extra, 'UseParallel')
                opts.UseParallel = ctrl.extra.UseParallel;
            end

            function chi2 = obj_fun(p_row)
                f = eval_fn(p_row(:), false);
                r = wt .* (yval - f);
                chi2 = r' * r;
            end

            [p_best_row, fval, exitflag] = particleswarm(@obj_fun, npfree, lb(:)', ub(:)', opts);
            p_best = p_best_row(:);
            chisqr_red = fval / nnorm;
            converged = (exitflag > 0);
            if ~converged
                warning('HERBERT:mf_fitter_particleswarm:not_converged', ...
                    'WARNING: particleswarm did not converge (exitflag = %d)', exitflag);
            end

            [sig, cor] = obj.estimate_covariance_(eval_fn, wt, p_best, chisqr_red, options);
        end
    end
end
