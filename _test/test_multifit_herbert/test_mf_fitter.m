classdef test_mf_fitter < TestCase
% Unit tests for replaceable mf_fitter_base optimization algorithms

    methods
        function obj = test_mf_fitter(name)
            if nargin < 1
                name = 'test_mf_fitter';
            end
            obj = obj@TestCase(name);
        end

        function test_default_fitter_is_lsqr(~)
            ds = IX_dataset_1d(1:5, 2*(1:5) + 1, ones(1, 5));
            kk = mfclass(ds);
            assertTrue(isa(kk.fitter, 'mf_fitter_lsqr'));
            assertEqual(kk.fitter.name, 'lsqr');
        end

        function test_set_fitter_validation(~)
            kk = mfclass();
            assertExceptionThrown(@() kk.set_fitter('unknown'), ...
                'HERBERT:mfclass:invalid_argument');
            assertExceptionThrown(@() kk.set_fitter(42), ...
                'HERBERT:mfclass:invalid_argument');
            assertExceptionThrown(@() kk.set_fitter([mf_fitter_lsqr(), mf_fitter_lsqr()]), ...
                'HERBERT:mfclass:invalid_argument');

            kk = kk.set_fitter(mock_mf_fitter());
            assertTrue(isa(kk.fitter, 'mock_mf_fitter'));
            kk = kk.set_fitter('-default');
            assertTrue(isa(kk.fitter, 'mf_fitter_lsqr'));
        end

        function test_custom_fitter_matches_lsqr_on_xye(~)
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
    end
end
