classdef tobyfit_DGdisk_res_model < tobyfit_mc_res_model_base
% Monte Carlo resolution model for double-disk chopper direct-geometry spectrometers
%
% Implements instrument-specific lookup tables, Jacobian dq_matrix_DGdisk,
% moderator/shaping/monochromating chopper and divergence sampling, and
% beamline covariance assembly for IX_inst_DGdisk instruments.

    properties (Constant)
        name = 'DGdisk'
    end

    properties (Constant, Access = protected)
        use_tubes_ = true
    end

    methods
        function names = mc_contribution_names(~)
            % Return cellstr array of valid Monte Carlo contribution options.
            names = {'moderator', 'shape_chopper', 'mono_chopper', ...
                'horiz_divergence', 'vert_divergence', 'sample', ...
                'detector_depth', 'detector_area', 'energy_bin', 'mosaic'};
        end
    end

    methods (Access = protected)
        function [ei, x0, xa, x1, inst_data] = init_instrument_data_(~, header)
            [ei, x0, xa, x1, mod_shape_mono, horiz_div, vert_div] = ...
                instpars_DGdisk(header);
            inst_data = struct( ...
                'mod_shape_mono', {mod_shape_mono}, ...
                'horiz_div', {horiz_div}, ...
                'vert_div', {vert_div});
        end

        function lookup = pack_instrument_lookup_(~, lookup, inst_data, build_tables)
            nw = numel(inst_data);
            mod_shape_mono = cell(nw, 1);
            horiz_div = cell(nw, 1);
            vert_div = cell(nw, 1);
            for iw = 1:nw
                mod_shape_mono{iw} = inst_data{iw}.mod_shape_mono;
                horiz_div{iw} = inst_data{iw}.horiz_div;
                vert_div{iw} = inst_data{iw}.vert_div;
            end
            if build_tables
                lookup.mod_shape_mono_table = object_lookup(mod_shape_mono);
                lookup.horiz_div_table = object_lookup(horiz_div);
                lookup.vert_div_table = object_lookup(vert_div);
            end
        end

        function dq_mat = compute_dq_matrix_(~, lookup, iw, irun, kf, x2, ...
                s_mat_run, f_mat, d_mat, spec_to_rlu_run)
            ki = lookup.ki{iw};
            xa = lookup.xa{iw};
            x1 = lookup.x1{iw};
            dq_mat = dq_matrix_DGdisk(ki(irun), kf, ...
                xa(irun), x1(irun), x2, ...
                s_mat_run, f_mat, d_mat, spec_to_rlu_run, ...
                lookup.k_to_v, lookup.k_to_e);
        end

        function yvec_beam = sample_beamline_deviations_(~, lookup, iw, irun, ...
                npix, mc_contributions)
            yvec_beam = zeros(4, 1, npix);
            mc_mod_shape_mono = [mc_contributions.moderator, ...
                mc_contributions.shape_chopper, mc_contributions.mono_chopper];

            tchop_av = lookup.mod_shape_mono_table.func_eval_ind( ...
                iw, irun, @mean, mc_mod_shape_mono);
            yvec_beam([1, 4], 1, :) = 1e-6 * ...
                (lookup.mod_shape_mono_table.rand_ind(iw, irun, @rand, 'mc', mc_mod_shape_mono) - tchop_av);
            debugtools(@debug_histogram_array, (10^6) * yvec_beam(1, 1, :), 't_shape', 'microseconds');

            if mc_contributions.horiz_divergence
                yvec_beam(2, 1, :) = lookup.horiz_div_table.rand_ind(iw, irun, @rand);
            end
            if mc_contributions.vert_divergence
                yvec_beam(3, 1, :) = lookup.vert_div_table.rand_ind(iw, irun, @rand);
            end
        end

        function cov_x = fill_beamline_covariance_(~, cov_x, lookup, iw, irun)
            cov_sh_ch = 1e-12 * lookup.mod_shape_mono_table.func_eval_ind(iw, irun, @covariance);
            var_horiz_div = (lookup.horiz_div_table.func_eval_ind(iw, irun, @profile_width)).^2;
            var_vert_div = (lookup.vert_div_table.func_eval_ind(iw, irun, @profile_width)).^2;
            cov_x([1, 4], [1, 4], :) = cov_sh_ch;
            cov_x(2, 2, :) = var_horiz_div;
            cov_x(3, 3, :) = var_vert_div;
        end

        function [lookup, pars_out] = update_moderator_lookup_(~, lookup, modshape, pars_in)
            moderator = lookup.mod_shape_mono_table.object_store(1).moderator;
            [moderator, pars_out] = refine_moderator_strip_pars(moderator, modshape, pars_in);
            mod_shape_mono = lookup.mod_shape_mono_table.object_store;
            for k = 1:numel(mod_shape_mono)
                mod_shape_mono(k).moderator = moderator;
            end
            lookup.mod_shape_mono_table.object_store = mod_shape_mono;
        end
    end
end
