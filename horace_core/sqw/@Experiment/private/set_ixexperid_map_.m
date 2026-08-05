function obj = set_ixexperid_map_(obj,val)
%SET_ixexperid_map_ Check and set ixexperid_map, connecting run-id, describing the
% experiment and the number of the experiment information header in the
% list of all experiment descriptors
%
% Main part of ixexperid_map setter procedure
if isa(val,'containers.Map')
    obj.ixexperid_map_ = val;
    keys = val.keys;
    keys = [keys{:}];
elseif isa(val,'fast_map')
    keys = val.keys;
    val  = val.values;
    obj.ixexperid_map_ = containers.Map(keys,val);
elseif isnumeric(val) && numel(val) == obj.n_runs
    keys = val(:)';
else
    error('HORACE:Experiment:invalid_argument', ...
        ['input for ixexperid_map should be map, defining connection between ixeper-ids and headers(IX_experiment(s)),\n', ...
        ' describing these runs or array of runid-s to set.\n', ...
        ' In fact it is: %s'], ...
        class(val))
end
obj = set_runids_map_and_synchonize_headers_(obj,keys);
%
if obj.do_check_combo_arg_
    obj = check_combo_arg(obj);
end

