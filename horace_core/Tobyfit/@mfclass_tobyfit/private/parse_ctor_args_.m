function [mf_args, res_model, fitter] = parse_ctor_args_(varargin)
% Strip optional Tobyfit-specific keyword-value pairs before mfclass()
%
% Recognised trailing/embedded keyword-value pairs:
%   'res_model', <tobyfit_res_model_base | instrument | char>
%   'fitter',    <mf_fitter_base | char>

res_model = [];
fitter = [];
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
        end
    end
    i = i + 1;
end

mf_args = varargin(keep);
end
