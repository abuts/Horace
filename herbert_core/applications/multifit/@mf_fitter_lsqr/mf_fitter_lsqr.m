classdef mf_fitter_lsqr < mf_fitter_base
% Default Levenberg-Marquardt least-squares fitter for mfclass / Tobyfit
%
% Wraps Herbert's serial and parallel Levenberg-Marquardt solvers
% (multifit_lsqr_ser and multifit_lsqr_par), preserving finite-difference
% Jacobian calculation, RNG state caching, and parallel execution via
% hpc_config.

    properties (Constant)
        name = 'lsqr'
    end

    methods
        function obj = mf_fitter_lsqr()
            % Construct default Levenberg-Marquardt least-squares fitter.
        end

        function [p_best, sig, cor, chisqr_red, converged] = fit(~, w, xye, ...
                func, bfunc, pin, bpin, f_pass_caller_info, bf_pass_caller_info, ...
                pfin, p_info, options, perform_fit)
            % Run Levenberg-Marquardt least-squares minimization or chi2 evaluation.
            if nargin < 12 || ~isstruct(options)
                listing = 0;
                fcp = [0.0001, 20, 0.001];
            else
                listing = options.listing;
                fcp = options.fit_control_parameters;
            end
            if nargin < 13 || isempty(perform_fit)
                perform_fit = true;
            end

            [p_best, sig, cor, chisqr_red, converged] = multifit_lsqr( ...
                w, xye, func, bfunc, pin, bpin, ...
                f_pass_caller_info, bf_pass_caller_info, ...
                pfin, p_info, listing, fcp, perform_fit);
        end
    end
end
