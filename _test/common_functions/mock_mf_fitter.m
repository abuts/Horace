classdef mock_mf_fitter < mf_fitter_base
% Test fitter subclass using MATLAB fminsearch via mf_fitter_base.minimize_
%
% Verifies that custom mf_fitter_base subclasses can replace mf_fitter_lsqr
% on both mfclass and mfclass_tobyfit using only minimize_().

    properties (Constant)
        name = 'mock_fminsearch'
    end

    methods
        function obj = mock_mf_fitter()
        end
    end

    methods (Access = protected)
        function [p_best, sig, cor, chisqr_red, converged] = minimize_( ...
                ~, eval_fn, yval, wt, pfin, ~, ~)
            npfree = numel(pfin);
            nnorm = max(numel(yval) - npfree, 1);

            if npfree == 0
                f0 = eval_fn(pfin, false);
                r0 = wt .* (yval - f0);
                p_best = pfin;
                sig = zeros(1, 0);
                cor = zeros(0, 0);
                chisqr_red = (r0' * r0) / nnorm;
                converged = true;
                return
            end

            function chi2 = obj_fun(p)
                f = eval_fn(p, false);
                r = wt .* (yval - f);
                chi2 = r' * r;
            end

            opts = optimset('Display', 'off', 'TolX', 1e-6, 'TolFun', 1e-6);
            [p_best, fval, exitflag] = fminsearch(@obj_fun, pfin, opts);
            chisqr_red = fval / nnorm;
            converged = (exitflag == 1);
            sig = zeros(size(p_best));
            cor = eye(npfree);
        end
    end
end
