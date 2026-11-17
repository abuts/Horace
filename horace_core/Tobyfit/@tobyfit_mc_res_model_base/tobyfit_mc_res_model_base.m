classdef (Abstract) tobyfit_mc_res_model_base < tobyfit_res_model_base
% Shared Monte Carlo resolution convolution and covariance base class
%
% Implements the Template Method pattern for direct-geometry chopper
% spectrometer Monte Carlo resolution broadening, lookup table
% initialisation, and 4D (Q,w) covariance matrix computation.
%
% Subclasses (e.g. tobyfit_DGfermi_res_model, tobyfit_DGdisk_res_model)
% supply only the instrument-specific hooks:
%   init_instrument_data_
%   pack_instrument_lookup_
%   compute_dq_matrix_
%   sample_beamline_deviations_
%   fill_beamline_covariance_
%   update_moderator_lookup_

    properties (Abstract, Constant, Access = protected)
        use_tubes_      % Logical: use 3He tube geometry (true) or classic (false)
    end

    methods
        function [ok, mess, lookup, npix] = init(obj, win, varargin)
            % Precompute lookup tables and matrix transformations for Tobyfit.
            if nargin == 1
                ok = true;
                mess = '';
                lookup = obj.mc_contribution_names();
                npix = [];
                return
            end

            nw = numel(win);
            lookup = struct([]);
            if iscell(win)
                lookup = {lookup};
            end
            npix = zeros(size(win));

            keywrd_def = struct('tables', 1);
            flags = {'tables'};
            [args, keywrd] = parse_arguments(varargin, keywrd_def, flags);
            if isscalar(args)
                ipix = args{1};
            elseif ~isempty(args)
                ok = false;
                mess = 'Check the number of input arguments';
                return
            end

            all_pixels = ~exist('ipix', 'var');
            if ~all_pixels
                parse_pixel_indices(win, ipix);
            end

            c = neutron_constants;
            k_to_e = c.c_k_to_emev;
            k_to_v = 1e6 / c.c_t_to_k;
            deps_to_dt = 0.5e-6 * c.c_t_to_k / c.c_k_to_emev;

            ei = cell(nw, 1);
            x0 = cell(nw, 1);
            xa = cell(nw, 1);
            x1 = cell(nw, 1);
            ki = cell(nw, 1);
            kf = cell(nw, 1);
            sample = cell(nw, 1);
            s_mat = cell(nw, 1);
            spec_to_rlu = cell(nw, 1);
            alatt = cell(nw, 1);
            angdeg = cell(nw, 1);
            is_mosaic = cell(nw, 1);
            detectors = cell(nw, 1);
            dt = cell(nw, 1);
            en = cell(nw, 1);
            inst_data = cell(nw, 1);

            n_runs = NaN(nw, 1);
            for iw = 1:nw
                wtmp = get_sqw_item_(win, iw);
                n_runs(iw) = wtmp.experiment_info.n_runs;
                detectors{iw} = detector_array(wtmp, obj.use_tubes_);
            end
            sz_cell = num2cell([n_runs, ones(nw, 1)], 2);
            detector_table = object_lookup(detectors, 'repeat', sz_cell);

            for iw = 1:nw
                wtmp = get_sqw_item_(win, iw);
                if all_pixels
                    [irun, idet, ien] = parse_pixel_indices(wtmp);
                elseif iscell(ipix) && numel(ipix) > 1
                    [irun, idet, ien] = parse_pixel_indices(wtmp, ipix{iw});
                else
                    [irun, idet, ien] = parse_pixel_indices(wtmp, ipix);
                end
                npix(iw) = numel(irun);
                irun = irun(:);
                idet = idet(:);
                ien = ien(:);

                [deps, eps_lo, eps_hi, ne] = energy_transfer_info(wtmp.experiment_info);
                deps = deps(:);
                eps_lo = eps_lo(:);
                eps_hi = eps_hi(:);
                ne = ne(:);
                if isscalar(ne)
                    ne_run = repmat(ne, size(irun));
                else
                    ne_run = ne(irun);
                end
                eps = eps_lo(irun);
                multi = ne_run > 1;
                if any(multi)
                    ir_m = irun(multi);
                    eps(multi) = (eps_lo(ir_m) .* (ne_run(multi) - ien(multi)) + ...
                        eps_hi(ir_m) .* (ien(multi) - 1)) ./ (ne_run(multi) - 1);
                end

                [ei_i, x0_i, xa_i, x1_i, inst_data{iw}] = ...
                    obj.init_instrument_data_(wtmp.experiment_info);
                ei{iw} = ei_i(:);
                x0{iw} = x0_i(:);
                xa{iw} = xa_i(:);
                x1{iw} = x1_i(:);

                ki{iw} = sqrt(ei{iw} / k_to_e);
                kf_i = sqrt((ei{iw}(irun) - eps(:)) / k_to_e);
                kf{iw} = kf_i(:);

                [sample{iw}, s_mat{iw}, spec_to_rlu{iw}, alatt{iw}, angdeg{iw}] = ...
                    sample_coords_to_spec_to_rlu(wtmp.experiment_info);

                mosaic = arrayfun(@(x)(x.eta), sample{iw});
                is_mosaic{iw} = any(mosaic_crystal(mosaic));

                x2 = detector_table.func_eval_ind(iw, irun, idet, @detector_info);
                dt_i = deps_to_dt * (x2(:) .* deps(irun) ./ kf{iw}.^3);
                dt{iw} = dt_i(:);
                en{iw} = eps(:);
            end

            ok = true;
            mess = '';
            lookup = struct();

            if keywrd.tables
                lookup.sample_table = object_lookup(sample, 'repeat', sz_cell);
                lookup.detector_table = detector_table;
            end
            lookup.ei = ei;
            lookup.x0 = x0;
            lookup.xa = xa;
            lookup.x1 = x1;
            lookup.ki = ki;
            lookup.kf = kf;
            lookup.s_mat = s_mat;
            lookup.spec_to_rlu = spec_to_rlu;
            lookup.alatt = alatt;
            lookup.angdeg = angdeg;
            lookup.is_mosaic = is_mosaic;
            lookup.dt = dt;
            lookup.en = en;
            lookup.k_to_v = k_to_v;
            lookup.k_to_e = k_to_e;

            lookup = obj.pack_instrument_lookup_(lookup, inst_data, logical(keywrd.tables));

            if iscell(win)
                lookup = {lookup};
            end
        end

        function [wout, state_out, store_out] = convolve(obj, win, caller, ...
                state_in, ~, sqwfunc, pars, lookup, ...
                mc_contributions, mc_points, xtal, modshape)
            % Calculate Monte Carlo resolution-broadened sqw object(s).
            ind = caller.ind;
            if numel(ind) ~= numel(win)
                error('HORACE:tobyfit_resconv:invalid_argument', ...
                    'Inconsistency between number of input datasets and number passed from control routine');
            elseif numel(ind) ~= numel(state_in)
                error('HORACE:tobyfit_resconv:invalid_argument', ...
                    'Inconsistency between number of input datasets and number of internal function status stores');
            end

            refine_crystal = ~isempty(xtal);
            refine_moderator = ~isempty(modshape);
            if refine_crystal && refine_moderator
                error('HORACE:tobyfit_resconv:invalid_argument', ...
                    'Cannot refine both crystal and moderator parameters. Error in logic flow - this should have been caught');
            end

            wout = copy(win);
            state_out = cell(size(win));
            store_out = [];

            if ~iscell(pars)
                pars = {pars};
            end

            if refine_moderator
                [lookup, pars{1}] = obj.update_moderator_lookup_(lookup, modshape, pars{1});
            end

            reset_state = caller.reset_state;
            max_pix_size = get(hor_config, 'mem_chunk_size') / 10;

            for i = 1:numel(ind)
                iw = ind(i);

                if reset_state
                    if ~isempty(state_in{i})
                        rng(state_in{i});
                    end
                else
                    state_out{i} = rng;
                end

                pars_i = pars;
                if refine_crystal
                    [win(i), pars_i{1}] = refine_crystal_strip_pars(win(i), xtal, pars_i{1});
                    [~, lookup.s_mat{iw}, lookup.spec_to_rlu{iw}, ...
                        lookup.alatt{iw}, lookup.angdeg{iw}] = ...
                        sample_coords_to_spec_to_rlu(win(i).experiment_info);
                end

                npix = win(i).data.npix(:);
                npix_chunks = split_vector_fixed_sum(npix, max_pix_size);
                pix_bin_regions = [1, cumsum(cellfun(@sum, npix_chunks))];

                for j = 1:numel(pix_bin_regions) - 1
                    wout(i) = obj.compute_resconv_chunk_(wout(i), iw, lookup, ...
                        sqwfunc, pars_i, mc_contributions, mc_points, ...
                        pix_bin_regions(j), pix_bin_regions(j + 1));
                end
                wout(i) = recompute_bin_data(wout(i));
            end
        end

        function [cov_proj, cov_spec, cov_hkle] = covariance(obj, win, ipix)
            % Return 4D momentum-energy covariance matrices [4,4,npix] in proj, spec, hkle frames.
            if exist('ipix', 'var')
                all_pixels = false;
                [ok, mess, lookup, npix_arr] = obj.init(win, ipix);
            else
                all_pixels = true;
                [ok, mess, lookup, npix_arr] = obj.init(win);
            end
            if ~ok
                error('HORACE:tobyfit_resfun_covariance:invalid_argument', '%s', mess);
            end

            sample_table = lookup.sample_table;
            detector_table = lookup.detector_table;

            cov_proj = cell(size(win));
            cov_spec = cell(size(win));
            cov_hkle = cell(size(win));

            for iw = 1:numel(win)
                if all_pixels
                    [irun, idet] = parse_pixel_indices(win(iw));
                elseif iscell(ipix) && numel(ipix) > 1
                    [irun, idet] = parse_pixel_indices(win(iw), ipix{iw});
                else
                    [irun, idet] = parse_pixel_indices(win(iw), ipix);
                end
                irun = irun(:);
                idet = idet(:);
                npix = npix_arr(iw);

                kf = lookup.kf{iw};
                s_mat = lookup.s_mat{iw};
                spec_to_rlu = lookup.spec_to_rlu{iw};
                dt = lookup.dt{iw};

                [x2, ~, d_mat, f_mat] = detector_table.func_eval_ind(iw, irun, idet, @detector_info);
                dq_mat = obj.compute_dq_matrix_(lookup, iw, irun, kf, x2, ...
                    s_mat(:, :, irun), f_mat, d_mat, spec_to_rlu(:, :, irun));

                cov_x = zeros(11, 11, npix);
                cov_x = obj.fill_beamline_covariance_(cov_x, lookup, iw, irun);
                cov_x(5:7, 5:7, :) = sample_table.func_eval_ind(iw, irun, @covariance);
                cov_x(8:10, 8:10, :) = detector_table.func_eval_ind(iw, irun, idet, 'split', @covariance, kf);
                cov_x(11, 11, :) = dt.^2 / 12;

                if ~isa(win(iw).data.proj, 'LineProjBase')
                    error('HORACE:sqw:not_implemented', ...
                        'resolution cannot currently be calculated for any projection except linear projection');
                end

                cov_hkle{iw} = transform_matrix_(cov_x, dq_mat);
                cov_proj{iw} = transform_matrix_(cov_hkle{iw}, inv(win(iw).data.u_to_rlu));
                rlu_to_spec = invert_matrix_(spec_to_rlu);
                rlu_to_spec4 = zeros(4, 4, numel(irun));
                rlu_to_spec4(1:3, 1:3, :) = rlu_to_spec(:, :, irun);
                rlu_to_spec4(4, 4, :) = 1;
                cov_spec{iw} = transform_matrix_(cov_hkle{iw}, rlu_to_spec4);
            end

            if isscalar(win)
                cov_hkle = cov_hkle{1};
                cov_proj = cov_proj{1};
                cov_spec = cov_spec{1};
            end
        end
    end

    methods (Access = private)
        function sqw_obj = compute_resconv_chunk_(obj, sqw_obj, iw, lookup, ...
                sqwfunc, pars, mc_contributions, mc_points, idx_start, idx_end)
            sample_table = lookup.sample_table;
            detector_table = lookup.detector_table;

            ki = lookup.ki{iw};
            s_mat = lookup.s_mat{iw};
            spec_to_rlu = lookup.spec_to_rlu{iw};
            alatt = lookup.alatt{iw};
            angdeg = lookup.angdeg{iw};
            is_mosaic = lookup.is_mosaic{iw};

            kf = lookup.kf{iw}(idx_start:idx_end, :);
            dt = lookup.dt{iw}(idx_start:idx_end, :);
            en = lookup.en{iw}(idx_start:idx_end, :);

            [irun, idet] = parse_pixel_indices(sqw_obj, (idx_start:idx_end)');
            npix = idx_end - idx_start + 1;

            [x2, detdcn, d_mat, f_mat] = detector_table.func_eval_ind(iw, irun, idet, @detector_info);
            q = calculate_q_(ki(irun), kf, detdcn, spec_to_rlu(:, :, irun));

            dq_mat = obj.compute_dq_matrix_(lookup, iw, irun, kf, x2, ...
                s_mat(:, :, irun), f_mat, d_mat, spec_to_rlu(:, :, irun));

            stmp = zeros(npix, 1);
            for imc = 1:mc_points
                yvec = zeros(11, 1, npix);

                % Beamline deviations (yvec 1:4) supplied by instrument subclass
                yvec(1:4, 1, :) = obj.sample_beamline_deviations_( ...
                    lookup, iw, irun, npix, mc_contributions);

                % Sample deviations (yvec 5:7)
                if mc_contributions.sample
                    yvec(5:7, 1, :) = sample_table.rand_ind(iw, irun, @rand);
                end

                % Detector deviations (yvec 8:10)
                if mc_contributions.detector_depth || mc_contributions.detector_area
                    det_points = detector_table.rand_ind(iw, irun, idet, 'split', @rand, kf);
                    if ~mc_contributions.detector_area
                        yvec(8, 1, :) = det_points(1, :);
                    elseif ~mc_contributions.detector_depth
                        yvec(9:10, 1, :) = det_points(2:3, :);
                    else
                        yvec(8:10, 1, :) = det_points;
                    end
                end

                % Energy bin (yvec 11)
                if mc_contributions.energy_bin
                    yvec(11, 1, :) = dt' .* (rand(1, npix) - 0.5);
                end

                dq = mtimesx_horace(dq_mat, yvec);
                qw = dq + reshape([q{1}'; q{2}'; q{3}'; en'], size(dq));

                if is_mosaic && mc_contributions.mosaic
                    Rrlu = sample_table.rand_ind(iw, irun, @rand_mosaic, alatt, angdeg);
                    qw(1:3, :, :) = mtimesx_horace(Rrlu, qw(1:3, :, :));
                end
                qw = squeeze(qw);

                stmp = stmp + sqwfunc(qw(1, :)', qw(2, :)', qw(3, :)', qw(4, :)', pars{:});
            end

            sqw_obj.pix.signal(idx_start:idx_end) = stmp(:)' / mc_points;
            sqw_obj.pix.variance(idx_start:idx_end) = zeros(1, numel(stmp));
        end
    end

    methods (Abstract, Access = protected)
        [ei, x0, xa, x1, inst_data] = init_instrument_data_(obj, header)
        lookup = pack_instrument_lookup_(obj, lookup, inst_data, build_tables)
        dq_mat = compute_dq_matrix_(obj, lookup, iw, irun, kf, x2, s_mat_run, f_mat, d_mat, spec_to_rlu_run)
        yvec_beam = sample_beamline_deviations_(obj, lookup, iw, irun, npix, mc_contributions)
        cov_x = fill_beamline_covariance_(obj, cov_x, lookup, iw, irun)
        [lookup, pars_out] = update_moderator_lookup_(obj, lookup, modshape, pars_in)
    end
end

function witem = get_sqw_item_(win, iw)
if iscell(win)
    witem = win{iw};
else
    witem = win(iw);
end
end

function q = calculate_q_(ki, kf, detdcn, spec_to_rlu)
q = -kf' .* detdcn;
q(1, :) = ki' + q(1, :);
q = mtimesx_horace(spec_to_rlu, reshape(q, [3, 1, numel(ki)]));
q = squeeze(q);
q = num2cell(q', 1);
end

function Cout = transform_matrix_(C, B)
szC = size(C);
szB = size(B);
if numel(szC) == 2 && numel(szB) == 2
    Cout = B * C * B';
else
    if numel(szB) == 2
        tmp = mtimesx_horace(C, B');
    else
        tmp = mtimesx_horace(C, permute(B, [2, 1, 3:numel(szB)]));
    end
    Cout = mtimesx_horace(B, tmp);
end
end

function Cinv = invert_matrix_(C)
szC = size(C);
if numel(szC) == 2
    Cinv = inv(C);
else
    Cinv = zeros(size(C));
    for i = 1:prod(szC(3:end))
        Cinv(:, :, i) = inv(C(:, :, i));
    end
end
end
