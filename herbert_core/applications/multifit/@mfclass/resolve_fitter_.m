function fitter_obj = resolve_fitter_(obj, pfin, p_info)
% Resolve active mf_fitter_base instance from object state / hor_config,
% applying extra fitter parameters and resolved parameter bounds.

if obj.fitter_is_explicit_
    fitter_obj = obj.fitter_;
else
    cfg_fitter = mf_get_fitter();
    if strcmp(class(cfg_fitter), class(obj.fitter_))
        fitter_obj = obj.fitter_;
    else
        fitter_obj = cfg_fitter;
    end
end

if ~isempty(fieldnames(obj.fitter_params_))
    fitter_obj = fitter_obj.set_fitter_params(obj.fitter_params_);
end

if isempty(obj.fitting_ranges_)
    return
end

if nargin < 3
    fitter_obj = fitter_obj.set_bounds(obj.fitting_ranges_);
    return
end

lb_spec = obj.fitting_ranges_.lb(:);
ub_spec = obj.fitting_ranges_.ub(:);
n_spec = numel(lb_spec);
npfree = numel(pfin);

if n_spec == npfree && npfree == p_info.npptot
    lb_free = lb_spec;
    ub_free = ub_spec;
elseif n_spec == p_info.npptot
    lb_free = lb_spec(p_info.free);
    ub_free = ub_spec(p_info.free);
elseif n_spec == npfree
    lb_free = lb_spec;
    ub_free = ub_spec;
elseif n_spec == p_info.nptot
    lb_full = [lb_spec; -inf(p_info.nbptot, 1)];
    ub_full = [ub_spec;  inf(p_info.nbptot, 1)];
    lb_free = lb_full(p_info.free);
    ub_free = ub_full(p_info.free);
else
    error('HERBERT:mfclass:invalid_argument', ...
        ['Fitting ranges length (%d) does not match free parameters (%d), ', ...
         'foreground parameters (%d), or total parameters (%d)'], ...
        n_spec, npfree, p_info.nptot, p_info.npptot);
end

fitter_obj = fitter_obj.set_bounds(lb_free, ub_free);
end
