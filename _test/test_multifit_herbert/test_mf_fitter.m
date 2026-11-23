classdef test_mf_fitter < TestCase
% Unit tests for replaceable mf_fitter_base optimization algorithms,
% hor_config.fitting_method selection, fitting ranges, and extra parameters.

    methods
        function obj = test_mf_fitter(name)
            if nargin < 1
                name = 'test_mf_fitter';
            end
            obj = obj@TestCase(name);
        end

        function test_default_fitter_is_lsqr(~)
            clob = set_temporary_config_options(hor_config, 'fitting_method', 'lsqr'); %#ok<NASGU>
            ds = IX_dataset_1d(1:5, 2*(1:5) + 1, ones(1, 5));
            kk = mfclass(ds);
            assertTrue(isa(kk.fitter, 'mf_fitter_lsqr'));
            assertEqual(kk.fitter.name, 'lsqr');
        end

        function test_set_fitter_validation(~)
            clob = set_temporary_config_options(hor_config, 'fitting_method', 'lsqr'); %#ok<NASGU>
            kk = mfclass();
            assertExceptionThrown(@() kk.set_fitter('unknown'), ...
                'HERBERT:mfclass:invalid_argument');
            assertExceptionThrown(@() kk.set_fitter(42), ...
                'HERBERT:mfclass:invalid_argument');
            assertExceptionThrown(@() kk.set_fitter([mf_fitter_lsqr(), mf_fitter_lsqr()]), ...
                'HERBERT:mfclass:invalid_argument');

            kk = kk.set_fitter(mock_mf_fitter());
            assertTrue(isa(kk.fitter, 'mock_mf_fitter'));
            kk = kk.set_fitter('fminsearch');
            assertTrue(isa(kk.fitter, 'mf_fitter_fminsearch'));
            kk = kk.set_fitter('-default');
            assertTrue(isa(kk.fitter, 'mf_fitter_lsqr'));
        end

        function test_custom_fitter_matches_lsqr_on_xye(~)
            clob = set_temporary_config_options(hor_config, 'fitting_method', 'lsqr'); %#ok<NASGU>
            x = linspace(-5, 5, 25)';
            true_p = [10, 0.5, 1.2];
            y = true_p(1) * exp(-0.5 * ((x - true_p(2)) / true_p(3)).^2);
            e = 0.05 * ones(size(x));
            gauss_fn = @(xv, p) p(1) * exp(-0.5 * ((xv - p(2)) / p(3)).^2);

            kk_lsqr = mfclass(x, y, e);
            kk_lsqr = kk_lsqr.set_fun(gauss_fn, [8, 0.1, 1.5]);
            [~, fp_lsqr] = kk_lsqr.fit();

            kk_custom = kk_lsqr.set_fitter(mock_mf_fitter());
            assertEqual(kk_custom.fitter.name, 'mock_fminsearch');
            [~, fp_custom] = kk_custom.fit();

            assertTrue(fp_custom.converged);
            assertElementsAlmostEqual(fp_custom.p, fp_lsqr.p, 'absolute', 1e-3);
        end

        function test_fminsearch_unconstrained_and_bounded(~)
            clob = set_temporary_config_options(hor_config, 'fitting_method', 'lsqr'); %#ok<NASGU>
            x = linspace(-5, 5, 31)';
            true_p = [10, 0.5, 1.2];
            y = true_p(1) * exp(-0.5 * ((x - true_p(2)) / true_p(3)).^2);
            e = 0.05 * ones(size(x));
            gauss_fn = @(xv, p) p(1) * exp(-0.5 * ((xv - p(2)) / p(3)).^2);

            kk = mfclass(x, y, e);
            kk = kk.set_fun(gauss_fn, [7, 0.0, 1.5]);
            kk = kk.set_fitter('fminsearch');
            [~, fp_unb] = kk.fit();
            assertTrue(fp_unb.converged);
            assertElementsAlmostEqual(fp_unb.p, true_p, 'absolute', 1e-3);
            assertTrue(all(isfinite(fp_unb.sig)));

            % Cap amplitude <= 8.0 so bound actively constrains solution
            kk_b = kk.set_fitting_ranges([0, -2, 0.2], [8.0, 2, 5]);
            kk_b = kk_b.set_fitter_parameters('MaxFunEvals', 1200);
            [~, fp_b] = kk_b.fit();
            assertTrue(fp_b.p(1) <= 8.0 + 1e-12);
            assertTrue(fp_b.p(1) >= 7.5);
        end

        function test_lsqnonlin_unconstrained_and_bounded(~)
            if ~mf_fitter_lsqnonlin.is_available()
                return
            end
            clob = set_temporary_config_options(hor_config, 'fitting_method', 'lsqr'); %#ok<NASGU>
            x = linspace(-5, 5, 31)';
            true_p = [10, 0.5, 1.2];
            y = true_p(1) * exp(-0.5 * ((x - true_p(2)) / true_p(3)).^2);
            e = 0.05 * ones(size(x));
            gauss_fn = @(xv, p) p(1) * exp(-0.5 * ((xv - p(2)) / p(3)).^2);

            kk = mfclass(x, y, e);
            kk = kk.set_fun(gauss_fn, [7, 0.0, 1.5]);
            kk = kk.set_fitter('lsqnonlin');
            [~, fp] = kk.fit();
            assertTrue(fp.converged);
            assertElementsAlmostEqual(fp.p, true_p, 'absolute', 1e-3);
            assertTrue(all(isfinite(fp.sig)));

            kk_b = kk.set_fitting_ranges([0, 8.0; -2, 2; 0.2, 5]);
            [~, fp_b] = kk_b.fit();
            assertTrue(fp_b.converged);
            assertElementsAlmostEqual(fp_b.p(1), 8.0, 'absolute', 1e-4);
        end

        function test_particleswarm_requires_bounds_and_fits(~)
            if ~mf_fitter_particleswarm.is_available()
                return
            end
            clob = set_temporary_config_options(hor_config, 'fitting_method', 'lsqr'); %#ok<NASGU>
            rng_old = rng(42, 'twister');
            clob_rng = onCleanup(@() rng(rng_old)); %#ok<NASGU>

            x = linspace(-4, 4, 25)';
            true_p = [8, -0.5, 1.0];
            y = true_p(1) * exp(-0.5 * ((x - true_p(2)) / true_p(3)).^2);
            e = 0.1 * ones(size(x));
            gauss_fn = @(xv, p) p(1) * exp(-0.5 * ((xv - p(2)) / p(3)).^2);

            kk = mfclass(x, y, e);
            kk = kk.set_fun(gauss_fn, [5, 0.2, 1.5]);
            kk = kk.set_fitter('particleswarm');

            % Missing bounds must raise clear error
            assertExceptionThrown(@() kk.fit(), 'HERBERT:mf_fitter_base:missing_bounds');

            % Infinite bounds must also be rejected for particleswarm
            kk_inf = kk.set_fitting_ranges([0, -inf, 0.2], [20, 5, 5]);
            assertExceptionThrown(@() kk_inf.fit(), 'HERBERT:mf_fitter_base:missing_bounds');

            % Finite bounds succeed and match true optimum
            kk_ok = kk.set_fitting_ranges([2, 15; -2, 2; 0.3, 3.0]);
            kk_ok = kk_ok.set_fitter_parameters('SwarmSize', 25, 'MaxIterations', 250);
            [~, fp] = kk_ok.fit();
            assertTrue(fp.converged);
            assertElementsAlmostEqual(fp.p, true_p, 'absolute', 0.05);
        end

        function test_hor_config_selects_fitting_method(~)
            clob = set_temporary_config_options(hor_config, 'fitting_method', 'lsqr'); %#ok<NASGU>

            hc = hor_config();
            assertEqual(hc.fitting_method, 'lsqr');
            assertEqual(hc.fitting_method_options, ...
                {'lsqr', 'fminsearch', 'lsqnonlin', 'particleswarm'});

            % Invalid or ambiguous choices fail cleanly
            assertExceptionThrown(@() set(hor_config, 'fitting_method', 'bogus', '-buffer'), ...
                'HORACE:hor_config:invalid_argument');
            assertExceptionThrown(@() set(hor_config, 'fitting_method', 'lsq', '-buffer'), ...
                'HORACE:hor_config:invalid_argument');

            x = linspace(0, 10, 21)';
            y = 2.5 * x + 1.0;
            e = 0.1 * ones(size(x));
            lin_fn = @(xv, p) p(1) * xv + p(2);

            kk_auto = mfclass(x, y, e);
            kk_auto = kk_auto.set_fun(lin_fn, [1.0, 0.0]);
            assertTrue(isa(kk_auto.fitter, 'mf_fitter_lsqr'));

            % Explicit override stays pinned even when hor_config changes
            kk_pinned = kk_auto.set_fitter(mock_mf_fitter());

            % Changing hor_config switches non-explicit object's fitter dynamically
            set(hor_config, 'fitting_method', 'fminsearch', '-buffer');
            assertTrue(isa(kk_auto.fitter, 'mf_fitter_fminsearch'));
            assertTrue(isa(kk_pinned.fitter, 'mock_mf_fitter'));

            [~, fp] = kk_auto.fit();
            assertTrue(fp.converged);
            assertElementsAlmostEqual(fp.p, [2.5, 1.0], 'absolute', 1e-3);

            % Resetting explicit object restores hor_config tracking
            kk_pinned = kk_pinned.set_fitter('-default');
            assertTrue(isa(kk_pinned.fitter, 'mf_fitter_fminsearch'));
        end

        function test_fitting_ranges_mapping_with_fixed_and_bg_params(~)
            clob = set_temporary_config_options(hor_config, 'fitting_method', 'fminsearch'); %#ok<NASGU>

            x = linspace(-4, 4, 30)';
            % Peak amp=10 (free), x0=0 (fixed), sig=1 (free); bg const=2 (free), grad=0 (fixed)
            y = 10 * exp(-0.5 * (x / 1.0).^2) + 2.0;
            e = 0.05 * ones(size(x));
            gauss_fn = @(xv, p) p(1) * exp(-0.5 * ((xv - p(2)) / p(3)).^2);
            lin_fn = @(xv, p) p(1) + p(2) * xv;

            kk = mfclass(x, y, e);
            kk = kk.set_fun(gauss_fn, [6, 0, 1.5]);
            kk = kk.set_free([1, 0, 1]);
            kk = kk.set_bfun(lin_fn, [1.0, 0.0]);
            kk = kk.set_bfree([1, 0]);

            % Supply ranges for all 5 (fore + back) parameters:
            kk_full = kk.set_fitting_ranges([0, 20; -1, 1; 0.2, 5; -5, 10; -1, 1]);
            [~, fp_full] = kk_full.fit();
            assertTrue(fp_full.converged);
            assertElementsAlmostEqual(fp_full.p, [10, 0, 1], 'absolute', 1e-2);
            assertElementsAlmostEqual(fp_full.bp, [2, 0], 'absolute', 1e-2);

            % Supply ranges only for the 3 free parameters:
            kk_free = kk.set_fitting_ranges([0, 20; 0.2, 5; -5, 10]);
            [~, fp_free] = kk_free.fit();
            assertTrue(fp_free.converged);
            assertElementsAlmostEqual(fp_free.p, [10, 0, 1], 'absolute', 1e-2);
        end
    end
end
