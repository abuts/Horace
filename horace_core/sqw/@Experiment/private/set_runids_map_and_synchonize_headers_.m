function  obj = set_runids_map_and_synchonize_headers_(obj,ixexperid_map_keys)
% set runid to the headers(IX_experiments) and synchronize these
% runid-s with runid map if this map exist. Build runid map if it does not
% exist
%
% Inputs:
% ixexperid_map_keys -- indices of IX_experiment of the runs to use as the
% keys of the runid map
%

if obj.n_runs ~= numel(ixexperid_map_keys)
    error('HORACE:Experiment:invalid_arguent', ...
        'number of elements in ixeperid map (%d) is no equal to number of IX_experiments stored in Expriment (%d)', ...
        numel(ixexperid_map_keys),obj.n_runs)
end
exp = obj.expdata;

if ~isempty(obj.ixexperid_map_)
    indxes = obj.ixexperid_map_.values;
    indxes  = [indxes{:}];
else
    indxes = 1:obj.n_runs;
    obj.ixexperid_map_ = containers.Map(ixexperid_map_keys,indxes);
end
for i=1:numel(indxes)
    exp(indxes(i)).ixexper_id = ixexperid_map_keys(i);
end
obj.expdata_ = exp;
obj.ixexperid_map_ = containers.Map(ixexperid_map_keys,indxes);

