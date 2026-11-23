function [fitter, method_name] = mf_get_fitter(val)
% Resolve fitting algorithm from hor_config, name, or mf_fitter_base object
%
%   >> fitter = mf_get_fitter()          % Uses hor_config().fitting_method (or 'lsqr')
%   >> fitter = mf_get_fitter('-default')% Uses hor_config().fitting_method (or 'lsqr')
%   >> fitter = mf_get_fitter(name)      % 'lsqr', 'fminsearch', 'lsqnonlin', 'particleswarm'
%   >> fitter = mf_get_fitter(fitter_obj)% Returns scalar mf_fitter_base unchanged
%
% Output:
% -------
%   fitter       Scalar instance of an mf_fitter_base subclass.
%   method_name  Canonical short name of the resolved fitter.

if nargin < 1 || isempty(val)
    method_name = get_configured_fitting_method_();
    fitter = instantiate_fitter_(method_name);
    return
end

if isa(val, 'mf_fitter_base')
    if ~isscalar(val)
        error('HERBERT:mf_get_fitter:invalid_argument', ...
            'Fitter object must be a scalar mf_fitter_base instance');
    end
    fitter = val;
    method_name = fitter.name;
    return
end

if ischar(val) || (isstring(val) && isscalar(val))
    key = lower(char(val));
    if strcmp(key, '-default') || strcmp(key, 'default') || ...
            (numel(key) >= 2 && strncmp(key, '-default', numel(key)))
        method_name = get_configured_fitting_method_();
        fitter = instantiate_fitter_(method_name);
        return
    end
    [fitter, method_name] = instantiate_fitter_(key);
    return
end

error('HERBERT:mf_get_fitter:invalid_argument', ...
    ['Fitter must be an mf_fitter_base instance or a valid fitter name ', ...
     '(''lsqr'', ''fminsearch'', ''lsqnonlin'', ''particleswarm'', ''-default'')']);
end

% -------------------------------------------------------------------------
function method_name = get_configured_fitting_method_()
method_name = 'lsqr';
if ~isempty(which('hor_config'))
    try
        cs = config_store.instance();
        hc = hor_config();
        if cs.is_configured(hc)
            cfg_val = cs.get_config_field(hc, 'fitting_method');
            if (ischar(cfg_val) || (isstring(cfg_val) && isscalar(cfg_val))) && ~isempty(char(cfg_val))
                method_name = lower(char(cfg_val));
            end
        end
    catch
        method_name = 'lsqr';
    end
end
end

% -------------------------------------------------------------------------
function [fitter, canonical_name] = instantiate_fitter_(key)
switch lower(char(key))
    case 'lsqr'
        fitter = mf_fitter_lsqr();
        canonical_name = 'lsqr';
    case {'fminsearch', 'simplex', 'nelder_mead', 'nelder-mead'}
        fitter = mf_fitter_fminsearch();
        canonical_name = 'fminsearch';
    case {'lsqnonlin', 'trust_region', 'trust-region'}
        if ~mf_fitter_lsqnonlin.is_available()
            error('HERBERT:mf_get_fitter:not_available', ...
                'Fitter ''lsqnonlin'' requires Optimization Toolbox (lsqnonlin not found)');
        end
        fitter = mf_fitter_lsqnonlin();
        canonical_name = 'lsqnonlin';
    case {'particleswarm', 'swarm'}
        if ~mf_fitter_particleswarm.is_available()
            error('HERBERT:mf_get_fitter:not_available', ...
                'Fitter ''particleswarm'' requires Global Optimization Toolbox (particleswarm not found)');
        end
        fitter = mf_fitter_particleswarm();
        canonical_name = 'particleswarm';
    otherwise
        error('HERBERT:mf_get_fitter:invalid_argument', ...
            ['Unrecognised fitter name "%s". Valid options are: ', ...
             '''lsqr'', ''fminsearch'', ''lsqnonlin'', ''particleswarm'', ''-default''.'], ...
            char(key));
end
end
