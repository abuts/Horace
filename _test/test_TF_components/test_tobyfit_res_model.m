classdef test_tobyfit_res_model < TestCase
% Unit tests for replaceable Tobyfit resolution models, fitters, and helpers

    properties
        w2a
    end

    methods
        function obj = test_tobyfit_res_model(name)
            if nargin < 1
                name = 'test_tobyfit_res_model';
            end
            obj = obj@TestCase(name);

            ld = load('test_tobyfit_resfun_data.mat', 'w2a');
            sample = IX_sample(true, [1, 0, 0], [0, 1, 0], 'cuboid', [0.02, 0.02, 0.02]);
            sample.alatt = [4.2240, 4.2240, 4.2240];
            sample.angdeg = [90, 90, 90];
            obj.w2a = set_sample_and_inst(ld.w2a, sample, ...
                @maps_instrument_obj_for_tests, '-efix', 300, 'S');
        end

        function test_get_res_model_factory(obj)
            inst_fermi = obj.w2a.experiment_info.instruments{1};
            m_fermi = tobyfit_get_res_model(inst_fermi);
            assertTrue(isa(m_fermi, 'tobyfit_DGfermi_res_model'));

            m_disk = tobyfit_get_res_model('IX_inst_DGdisk');
            assertTrue(isa(m_disk, 'tobyfit_DGdisk_res_model'));

            mock = mock_tobyfit_res_model(1.5, 2.0);
            assertEqual(tobyfit_get_res_model(mock), mock);

            assertExceptionThrown(@() tobyfit_get_res_model('IX_null_inst'), ...
                'HORACE:tobyfit_get_res_model:invalid_argument');
        end

        function test_replace_resolution_model_in_tobyfit(obj)
            kk = tobyfit(obj.w2a);
            assertTrue(isa(kk.res_model, 'tobyfit_DGfermi_res_model'));

            mock_model = mock_tobyfit_res_model(2.5, 3.0);
            kk = kk.set_resolution_model(mock_model);
            assertTrue(isa(kk.res_model, 'mock_tobyfit_res_model'));
            assertEqual(fieldnames(kk.mc_contributions), {'mock_shift'; 'mock_scale'});

            % Test toggling mock contributions resets shift while keeping scale
            kk = kk.set_mc_contributions('mock_scale');
            kk = kk.set_fun(@(qh, qk, ql, en, p) p(1) + p(2) * en, [1, 2]);
            [w_noshift, ~] = kk.simulate();

            kk = kk.set_mc_contributions('all');
            [w_shift, ~] = kk.simulate();

            % Expect uniform difference of 3.0 * 2.0 * 2.5 = 15 on populated pixels
            diff_sig = w_shift.pix.signal - w_noshift.pix.signal;
            assertElementsAlmostEqual(diff_sig, repmat(15, size(diff_sig)), ...
                'absolute', 1e-10);
        end

        function test_replace_fitter_in_tobyfit(obj)
            kk = tobyfit(obj.w2a);
            kk = kk.set_resolution_model(mock_tobyfit_res_model(0, 1));
            kk = kk.set_fun(@(qh, qk, ql, en, p) p(1) + 0 * en, 10);
            [~, fp_lsqr] = kk.fit();

            kk = kk.set_fitter(mock_mf_fitter());
            [~, fp_mock] = kk.fit();

            assertEqual(kk.fitter.name, 'mock_fminsearch');
            assertTrue(fp_mock.converged);
            assertElementsAlmostEqual(fp_mock.p, fp_lsqr.p, 'absolute', 1e-3);
        end

        function test_legacy_wrappers_match_model_classes(obj)
            w = obj.w2a;
            m = tobyfit_DGfermi_res_model();
            [ok1, mess1, lkp1, npix1] = m.init(w);
            [ok2, mess2, lkp2, npix2] = tobyfit_DGfermi_resconv_init(w);
            assertTrue(ok1 && ok2);
            assertEqual(mess1, mess2);
            assertEqual(npix1, npix2);
            assertEqual(lkp1.kf, lkp2.kf);
            assertEqual(lkp1.dt, lkp2.dt);

            [cp1, cs1, ch1] = m.covariance(w, [1, 5, 10]);
            [cp2, cs2, ch2] = tobyfit_DGfermi_resfun_covariance(w, [1, 5, 10]);
            assertElementsAlmostEqual(cp1, cp2, 'absolute', 1e-12);
            assertElementsAlmostEqual(cs1, cs2, 'absolute', 1e-12);
            assertElementsAlmostEqual(ch1, ch2, 'absolute', 1e-12);
        end

        function test_refine_moderator_option_shorthand(obj)
            winc = obj.w2a;
            [pulse_model, ppmod] = get_mod_pulse(winc);
            kk = tobyfit(winc);
            kk = kk.set_refine_moderator([1, 0, 0]);
            assertEqual(kk.refine_moderator.pulse_model, pulse_model);
            assertEqual(kk.refine_moderator.pin, ppmod);
            assertEqual(kk.refine_moderator.free, [true, false, false]);

            kk = kk.set_refine_moderator(ppmod, [0, 1, 0]);
            assertEqual(kk.refine_moderator.free, [false, true, false]);
        end

        function test_resolution_plot_with_custom_model(obj)
            [cp_def, cs_def, ch_def, kept_def] = resolution_plot(obj.w2a, ...
                [0.3, 6; 0.7, 6], 'noplot');
            [cp_mock, ~, ~, kept_mock] = resolution_plot(obj.w2a, ...
                [0.3, 6; 0.7, 6], 'noplot', 'res_model', mock_tobyfit_res_model(0.2, 1));
            assertEqual(kept_def, kept_mock);
            assertEqual(size(cp_def), size(cp_mock));
            assertFalse(isequal(cp_def, cp_mock));
            assertEqual(size(cs_def, 1), 4);
            assertEqual(size(ch_def, 1), 4);
        end

        function test_tobyfit_constructor_keyword_injection(obj)
            mock_model = mock_tobyfit_res_model(1.0, 2.0);
            mock_fitter = mock_mf_fitter();
            kk = tobyfit(obj.w2a, 'res_model', mock_model, 'fitter', mock_fitter);
            assertTrue(isa(kk.res_model, 'mock_tobyfit_res_model'));
            assertEqual(kk.fitter.name, 'mock_fminsearch');

            kk = kk.set_fun(@(qh, qk, ql, en, p) p(1) + 0 * en, 5);
            [~, fp] = kk.fit();
            assertTrue(fp.converged);
        end

        function test_hor_config_and_fitting_ranges_in_tobyfit(obj)
            clob_cfg = set_temporary_config_options(hor_config, 'fitting_method', 'fminsearch'); %#ok<NASGU>
            clob_warn = set_temporary_warning('off', 'HERBERT:mask_data_for_fit:bad_points'); %#ok<NASGU>

            kk = tobyfit(obj.w2a, 'res_model', mock_tobyfit_res_model(0, 1));
            assertTrue(isa(kk.fitter, 'mf_fitter_fminsearch'));

            kk = kk.set_fun(@(qh, qk, ql, en, p) p(1) + 0 * en, 10);
            kk = kk.set_fitting_ranges([0, 50]);
            [~, fp_fmin] = kk.fit();
            assertTrue(fp_fmin.converged);
            assertTrue(fp_fmin.sig(1) > 0);
            assertTrue(fp_fmin.p(1) >= 0 && fp_fmin.p(1) <= 50);

            if mf_fitter_particleswarm.is_available()
                rng_old = rng(7, 'twister');
                clob_rng = onCleanup(@() rng(rng_old)); %#ok<NASGU>

                set(hor_config, 'fitting_method', 'particleswarm', '-buffer');
                assertTrue(isa(kk.fitter, 'mf_fitter_particleswarm'));

                kk_nobnd = kk.set_fitting_ranges([]);
                assertExceptionThrown(@() kk_nobnd.fit(), ...
                    'HERBERT:mf_fitter_base:missing_bounds');

                kk_pso = kk.set_fitting_ranges([0, 30]);
                kk_pso = kk_pso.set_fitter_parameters('SwarmSize', 15, 'MaxIterations', 100);
                [~, fp_pso] = kk_pso.fit();
                assertTrue(fp_pso.converged);
                assertTrue(fp_pso.p(1) >= 0 && fp_pso.p(1) <= 30);
                assertElementsAlmostEqual(fp_pso.p, fp_fmin.p, 'absolute', 1e-2);
            end
        end

        function test_tobyfit_real_mc_with_non_lsqr_and_moderator_ranges(obj)
            clob_cfg = set_temporary_config_options(hor_config, 'fitting_method', 'fminsearch'); %#ok<NASGU>
            clob_warn = set_temporary_warning('off', 'HERBERT:mask_data_for_fit:bad_points'); %#ok<NASGU>
            rng_old = rng(101, 'twister');
            clob_rng = onCleanup(@() rng(rng_old)); %#ok<NASGU>

            kk = tobyfit(obj.w2a);
            kk = kk.set_mc_points(1);
            kk = kk.set_fun(@(qh, qk, ql, en, p) p(1) + 0 * en, 10);
            kk = kk.set_fitting_ranges([0, 500]);
            kk = kk.set_fitter_parameters('MaxFunEvals', 80);

            if mf_fitter_lsqnonlin.is_available()
                set(hor_config, 'fitting_method', 'lsqnonlin', '-buffer');
                kk = kk.set_refine_moderator([1, 0, 0]);
                kk = kk.set_fitter_parameters('MaxIterations', 5);
                [~, fp, ok, ~, pulse_model, mod_p, mod_sig] = kk.fit();
                assertTrue(ok);
                assertEqual(numel(mod_p), 3);
                assertEqual(numel(mod_sig), 3);
                assertFalse(isempty(pulse_model));
                assertTrue(fp.p(1) >= 0 && fp.p(1) <= 500);
            else
                [~, fp] = kk.fit();
                assertTrue(fp.p(1) >= 0 && fp.p(1) <= 500);
            end
        end
    end
end
