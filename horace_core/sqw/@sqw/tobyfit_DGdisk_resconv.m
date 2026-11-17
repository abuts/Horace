function [wout, state_out, store_out] = tobyfit_DGdisk_resconv (win, caller, ...
    state_in, store_in, sqwfunc, pars, lookup, mc_contributions, mc_points, xtal, modshape)
% Deprecated thin wrapper delegating to tobyfit_DGdisk_res_model.convolve
%
% See also tobyfit_DGdisk_res_model

res_model = tobyfit_DGdisk_res_model();
[wout, state_out, store_out] = res_model.convolve(win, caller, ...
    state_in, store_in, sqwfunc, pars, lookup, mc_contributions, mc_points, xtal, modshape);
end
