classdef tobyfit_DGfermi_res_model < tobyfit_mc_res_model_base
% Monte Carlo resolution model for Fermi chopper direct-geometry spectrometers
%
% Implements instrument-specific lookup tables, Jacobian dq_matrix_DGfermi,
% moderator/aperture/Fermi chopper Monte Carlo sampling, and beamline
% covariance assembly for IX_inst_DGfermi instruments.

    properties (Constant)
        name = 'DGfermi'
    end

    properties (Constant, Access = protected)
        use_tubes_ = false
    end

    methods
        function names = mc_contribution_names(~)
            % Return cellstr array of valid Monte Carlo contribution options.
            names = {'moderator', 'aperture', 'chopper', 'sample', ...
                'detector_depth', 'detector_area', 'energy_bin', 'mosaic'};
        end
    end

    methods (Access = protected)
        function [ei, x0, xa, x1, inst_data] = init_instrument_data_(~, header)
            [ei, x0, xa, x1, thetam, angvel, moderator, aperture, chopper] = ...
                instpars_DGfermi(header);
            inst_data = struct( ...
                'thetam', thetam, ...
                'angvel', angvel, ...
                'moderator', {moderator}, ...
                'aperture', {aperture}, ...
                'chopper', {chopper});
        end

        function lookup = pack_instrument_lookup_(~, lookup, inst_data, build_tables)
            nw = numel(inst_data);
            thetam = cell(nw, 1);
            angvel = cell(nw, 1);
            moderator = cell(nw, 1);
            aperture = cell(nw, 1);
            chopper = cell(nw, 1);
            for iw = 1:nw
                thetam{iw} = inst_data{iw}.thetam(:);
                angvel{iw} = inst_data{iw}.angvel(:);
                moderator{iw} = inst_data{iw}.moderator;
                aperture{iw} = inst_data{iw}.aperture;
                chopper{iw} = inst_data{iw}.chopper;
            end
            if build_tables
                lookup.moderator_table = object_lookup(moderator);
                lookup.aperture_table = object_lookup(aperture);
                lookup.fermi_table = object_lookup(chopper);
            end
            lookup.thetam = thetam;
            lookup.angvel = angvel;
        end

        function dq_mat = compute_dq_matrix_(~, lookup, iw, irun, kf, x2, ...
                s_mat_run, f_mat, d_mat, spec_to_rlu_run)
            ki = lookup.ki{iw};
            x0 = lookup.x0{iw};
            xa = lookup.xa{iw};
            x1 = lookup.x1{iw};
            thetam = lookup.thetam{iw};
            angvel = lookup.angvel{iw};
            dq_mat = dq_matrix_DGfermi(ki(irun), kf, ...
                x0(irun), xa(irun), x1(irun), x2, thetam(irun), angvel(irun), ...
                s_mat_run, f_mat, d_mat, spec_to_rlu_run, ...
                lookup.k_to_v, lookup.k_to_e);
        end

        function yvec_beam = sample_beamline_deviations_(~, lookup, iw, irun, ...
                npix, mc_contributions)
            yvec_beam = zeros(4, 1, npix);
            if mc_contributions.moderator
                [~, mod_t_av] = lookup.moderator_table.func_eval_ind(iw, irun, @pulse_width);
                yvec_beam(1, 1, :) = 1e-6 * (lookup.moderator_table.rand_ind(iw, irun, @rand) - mod_t_av);
            end
            if mc_contributions.aperture
                yvec_beam(2:3, 1, :) = lookup.aperture_table.rand_ind(iw, irun, @rand);
            end
            if mc_contributions.chopper
                yvec_beam(4, 1, :) = 1e-6 * lookup.fermi_table.rand_ind(iw, irun, @rand);
            end
        end

        function cov_x = fill_beamline_covariance_(~, cov_x, lookup, iw, irun)
            cov_x(1, 1, :) = (1e-6 * lookup.moderator_table.func_eval_ind(iw, irun, @pulse_width)).^2;
            cov_x(2:3, 2:3, :) = lookup.aperture_table.func_eval_ind(iw, irun, @covariance);
            cov_x(4, 4, :) = (1e-6 * lookup.fermi_table.func_eval_ind(iw, irun, @pulse_width)).^2;
        end

        function [lookup, pars_out] = update_moderator_lookup_(~, lookup, modshape, pars_in)
            moderator = lookup.moderator_table.object_store(1);
            [moderator, pars_out] = refine_moderator_strip_pars(moderator, modshape, pars_in);
            lookup.moderator_table.object_store = moderator;
        end
    end
end
