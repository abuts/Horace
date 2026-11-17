classdef (Abstract) tobyfit_res_model_base
% Abstract base class for Tobyfit resolution convolution and covariance models
%
% Subclasses encapsulate an instrument/physics resolution model (for example
% Monte Carlo convolution for Fermi or disk chopper spectrometers, or
% custom analytical/Gaussian models) as a replaceable dependency of
% mfclass_tobyfit and resolution_plot.
%
% Subclasses must define the constant `name` property and implement:
%   mc_contribution_names(obj)
%   init(obj, win, varargin)
%   convolve(obj, win, caller, state_in, store_in, sqwfunc, pars, lookup, ...
%            mc_contributions, mc_points, xtal, modshape)
%   covariance(obj, win, ipix)

    properties (Abstract, Constant)
        name    % Short identifier for the resolution model
    end

    methods (Abstract)
        % Return cellstr array of valid Monte Carlo contribution component names
        names = mc_contribution_names(obj)

        % Precompute lookup tables and kinematic data for sqw datasets win
        [ok, mess, lookup, npix] = init(obj, win, varargin)

        % Evaluate resolution-broadened signal for sqw datasets win
        [wout, state_out, store_out] = convolve(obj, win, caller, ...
            state_in, store_in, sqwfunc, pars, lookup, ...
            mc_contributions, mc_points, xtal, modshape)

        % Compute 4D (Q,w) resolution covariance matrices [4,4,npix]
        [cov_proj, cov_spec, cov_hkle] = covariance(obj, win, ipix)
    end
end
