function obj = set_fitter_parameters(obj, varargin)
% Set additional algorithm-specific parameters for the fitting optimizer
%
% Clear additional fitter parameters (and ranges if '-default'):
%   >> obj = obj.set_fitter_parameters
%   >> obj = obj.set_fitter_parameters([])
%   >> obj = obj.set_fitter_parameters('-default')
%
% Set parameters via name-value pairs or struct:
%   >> obj = obj.set_fitter_parameters('fitting_ranges', ranges, 'MaxIterations', 300)
%   >> obj = obj.set_fitter_parameters('SwarmSize', 40, 'MaxStallIterations', 15)
%   >> obj = obj.set_fitter_parameters(param_struct)
%
% Special keys:
%   'fitting_ranges' or 'bounds' - sets parameter bounds via set_fitting_ranges.
%   Other fields are forwarded to the active mf_fitter_base subclass options.

if numel(varargin) == 0 || (numel(varargin) == 1 && isempty(varargin{1}))
    obj.fitter_params_ = struct();
    if obj.fitter_is_explicit_
        obj.fitter_ = obj.fitter_.set_fitter_params([]);
    end
    return
end

if numel(varargin) == 1 && (ischar(varargin{1}) || (isstring(varargin{1}) && isscalar(varargin{1})))
    key = lower(char(varargin{1}));
    if strcmp(key, '-default') || strcmp(key, 'default') || ...
            (numel(key) >= 2 && strncmp(key, '-default', numel(key)))
        obj.fitter_params_ = struct();
        obj.fitting_ranges_ = [];
        if obj.fitter_is_explicit_
            obj.fitter_ = obj.fitter_.set_fitter_params('-default');
        end
        return
    end
end

if numel(varargin) == 1
    arg = varargin{1};
    if isstruct(arg) && isscalar(arg)
        s_in = arg;
    else
        error('HERBERT:mfclass:invalid_argument', ...
            'Fitter parameters must be a scalar struct, name-value pairs, or ''-default''');
    end
elseif mod(numel(varargin), 2) == 0
    s_in = struct();
    for k = 1:2:numel(varargin)
        fname = varargin{k};
        if ~(ischar(fname) || (isstring(fname) && isscalar(fname))) || isempty(char(fname))
            error('HERBERT:mfclass:invalid_argument', ...
                'Fitter parameter names must be non-empty strings');
        end
        s_in.(char(fname)) = varargin{k+1};
    end
else
    error('HERBERT:mfclass:invalid_argument', ...
        'Fitter parameters must be supplied as a struct or name-value pairs');
end

fnames = fieldnames(s_in);
for k = 1:numel(fnames)
    fn = fnames{k};
    if strcmpi(fn, 'bounds') || strcmpi(fn, 'fitting_ranges')
        obj = obj.set_fitting_ranges(s_in.(fn));
    else
        obj.fitter_params_.(fn) = s_in.(fn);
        if obj.fitter_is_explicit_
            obj.fitter_ = obj.fitter_.set_fitter_params(fn, s_in.(fn));
        end
    end
end
end
