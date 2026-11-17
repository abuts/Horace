classdef mock_tobyfit_res_model < tobyfit_res_model_base
% Deterministic mock resolution model for verifying Tobyfit DI
%
% Replaces Monte Carlo resolution convolution with a controllable constant
% energy shift (`delta_en`) and returns a diagonal covariance matrix.

    properties (Constant)
        name = 'mock_res_model'
    end

    properties
        delta_en = 0
        scale = 1
    end

    methods
        function obj = mock_tobyfit_res_model(delta_en, scale)
            if nargin >= 1 && ~isempty(delta_en)
                obj.delta_en = delta_en;
            end
            if nargin >= 2 && ~isempty(scale)
                obj.scale = scale;
            end
        end

        function names = mc_contribution_names(~)
            names = {'mock_shift', 'mock_scale'};
        end

        function [ok, mess, lookup, npix] = init(obj, win, varargin)
            if nargin == 1
                ok = true;
                mess = '';
                lookup = obj.mc_contribution_names();
                npix = [];
                return
            end
            nw = numel(win);
            npix = zeros(size(win));
            kf = cell(nw, 1);
            dt = cell(nw, 1);
            en = cell(nw, 1);
            for iw = 1:nw
                if iscell(win)
                    wtmp = win{iw};
                else
                    wtmp = win(iw);
                end
                n = wtmp.pix.num_pixels;
                npix(iw) = n;
                kf{iw} = zeros(n, 1);
                dt{iw} = zeros(n, 1);
                en{iw} = wtmp.pix.dE(:);
            end
            lookup = struct('kf', {kf}, 'dt', {dt}, 'en', {en});
            if iscell(win)
                lookup = {lookup};
            end
            ok = true;
            mess = '';
        end

        function [wout, state_out, store_out] = convolve(obj, win, caller, ...
                ~, ~, sqwfunc, pars, ~, mc_contributions, ~, ~, ~)
            if ~iscell(pars)
                pars = {pars};
            end
            wout = copy(win);
            state_out = cell(size(win));
            store_out = [];

            shift = 0;
            if isfield(mc_contributions, 'mock_shift') && mc_contributions.mock_shift
                shift = obj.delta_en;
            end
            factor = 1;
            if isfield(mc_contributions, 'mock_scale') && mc_contributions.mock_scale
                factor = obj.scale;
            end

            for i = 1:numel(caller.ind)
                q_hkle = wout(i).pix.coordinates;
                sig = factor * sqwfunc(q_hkle(1, :)', q_hkle(2, :)', ...
                    q_hkle(3, :)', q_hkle(4, :)' + shift, pars{:});
                wout(i).pix.signal = sig(:)';
                wout(i).pix.variance = zeros(1, numel(sig));
                wout(i) = recompute_bin_data(wout(i));
            end
        end

        function [cov_proj, cov_spec, cov_hkle] = covariance(obj, win, ipix)
            nw = numel(win);
            cov_proj = cell(size(win));
            cov_spec = cell(size(win));
            cov_hkle = cell(size(win));
            for iw = 1:nw
                if exist('ipix', 'var')
                    if iscell(ipix) && numel(ipix) > 1
                        np = numel(ipix{iw});
                    else
                        np = numel(ipix);
                    end
                else
                    np = win(iw).pix.num_pixels;
                end
                c = repmat(diag([0.01, 0.01, 0.01, max(obj.delta_en^2, 0.01)]), [1, 1, np]);
                cov_proj{iw} = c;
                cov_spec{iw} = c;
                cov_hkle{iw} = c;
            end
            if isscalar(win)
                cov_proj = cov_proj{1};
                cov_spec = cov_spec{1};
                cov_hkle = cov_hkle{1};
            end
        end
    end
end
