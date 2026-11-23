

1. New Fitting Methods (herbert_core/applications/multifit/)
Extended Base Strategy — mf_fitter_base.m:
Added bounds (set_bounds), fitter_params (set_fitter_params), supports_bounds, and requires_bounds properties.
Added an initial eval_model(pfin, true) call prior to minimize_ so Monte Carlo RNG state (S) and lookup tables (Store) are seeded deterministically for non-Levenberg-Marquardt optimizers.
Extracted shared resolve_bounds_, parse_control_options_, estimate_covariance_, and covariance_from_jacobian_ helpers.

Additional Fitters:
mf_fitter_fminsearch.m ('fminsearch' / 'simplex' / 'nelder_mead'): derivative-free Nelder-Mead optimizer using core MATLAB fminsearch, supporting optional parameter bounds and finite-difference SVD uncertainties.
mf_fitter_lsqnonlin.m ('lsqnonlin'): Optimization Toolbox trust-region nonlinear least-squares solver with native [lb, ub] bounds and Jacobian-derived sig/cor (with static is_available() check).
mf_fitter_particleswarm.m ('particleswarm'): Global Optimization Toolbox particle swarm optimizer that enforces finite fitting ranges (requires_bounds = true) and forwards options such as SwarmSize, MaxIterations, FunctionTolerance, and HybridFcn.
mf_fitter_lsqr.m: preserves multifit_lsqr behavior unchanged and warns (HERBERT:mf_fitter_lsqr:bounds_not_supported) if finite fitting ranges are supplied.
Fitter Factory — mf_get_fitter.m: resolves an mf_fitter_base instance, optimizer name, or hor_config().fitting_method (falling back to 'lsqr').

2. Configuration & Fitting-Range / Parameter Wiring
hor_config Selection — hor_config.m:
Added saveable property fitting_method (default 'lsqr') and read-only fitting_method_options ({'lsqr', 'fminsearch', 'lsqnonlin', 'particleswarm'}), validated via parse_char_options and solver availability checks.
mfclass Integration — mfclass.m, set_fitter.m, resolve_fitter_.m, @mfclass/fit.m:
Automatically follows hor_config().fitting_method unless explicitly overridden via set_fitter(...) (set_fitter('-default') restores config tracking).
Resolves fitting_ranges against free parameters (p_info.free) whether supplied for all parameters (npptot), foreground parameters (nptot), or independently free parameters (npfree).
mfclass_tobyfit & @sqw/tobyfit API — set_fitting_ranges.m, set_fitter_parameters.m, mfclass_tobyfit.m, @mfclass_tobyfit/fit.m, parse_ctor_args_.m, tobyfit.m:
Added set_fitting_ranges / fitting_ranges and set_fitter_parameters / fitter_parameters (plus 'fitting_ranges' and 'fitter_parameters' keyword options in tobyfit(...) / mfclass_tobyfit(...)).
Pads pre-refinement model fitting_ranges with [-Inf, Inf] when refine_crystal or refine_moderator appends refinement parameters.

3. Verification (MATLAB R2025b)

All unit and integration checks ran after the final edit via validate_horace:

test_multifit_herbert (including test_mf_fitter.m): PASSED (19 test cases ran, 2 pre-existing skips, 0 failures).
test_config: PASSED (29 test cases ran, 0 skipped, 0 failures).
test_TF_components/test_tobyfit_res_model (test_tobyfit_res_model.m): PASSED (9 test cases ran, 0 skipped, 0 failures).
test_TF_components/test_tobyfit_cuts: validate_horace reported PASSED across 11 test cases (1 pre-existing skip, 0 failures), with MATLAB exiting 0xc0000005 during process teardown after completion.
checkcode: no syntax errors across the 21 checked files (only pre-existing notices in hor_config.m, @mfclass/mfclass.m, @mfclass/fit.m, and minor ISCL/MSNU style notices in set_fitting_ranges.m, set_fitter_parameters.m, and the new test cases).