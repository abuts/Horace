function [mf_args, res_model, fitter, fitting_ranges, fitter_params] = parse_ctor_args_(varargin)
% Strip optional Tobyfit-specific keyword-value pairs before mfclass()
%
% Recognised trailing/embedded keyword-value pairs:
%   'res_model',         <tobyfit_res_model_base | instrument | char>
%   'fitter',            <mf_fitter_base | char>
%   'fitting_ranges',    <bounds matrix / struct>
%   'fitter_parameters', <struct>

res_model = [];
fitter = [];
fitting_ranges = [];
fitter_params = [];
narg = numel(varargin);
keep = true(1, narg);

i = 1;
while i <= narg
    arg = varargin{i};
    if ischar(arg) || isstring(arg)
        key = lower(char(arg));
        if strcmp(key, 'res_model')
            if i == narg
                error('HORACE:mfclass_tobyfit:invalid_argument', ...
                    'Option ''res_model'' must be followed by a resolution model or instrument identifier');
            end
            res_model = varargin{i+1};
            keep(i:i+1) = false;
            i = i + 2;
            continue
        elseif strcmp(key, 'fitter')
            if i == narg
                error('HORACE:mfclass_tobyfit:invalid_argument', ...
                    'Option ''fitter'' must be followed by an mf_fitter_base instance or fitter name');
            end
            fitter = varargin{i+1};
            keep(i:i+1) = false;
            i = i + 2;
            continue
        elseif strcmp(key, 'fitting_ranges')
            if i == narg
                error('HORACE:mfclass_tobyfit:invalid_argument', ...
                    'Option ''fitting_ranges'' must be followed by fitting bounds');
            end
            fitting_ranges = varargin{i+1};
            keep(i:i+1) = false;
            i = i + 2;
            continue
        elseif strcmp(key, 'fitter_parameters')
            if i == narg
                error('HORACE:mfclass_tobyfit:invalid_argument', ...
                    'Option ''fitter_parameters'' must be followed by a parameter structure');
            end
            fitter_params = varargin{i+1};
            keep(i:i+1) = false;
            i = i + 2;
            continue
        end
    end
    i = i + 1;
end

mf_args = varargin(keep);
end
