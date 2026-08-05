function obj = build_runid_map_(obj)
% Build map connecting pixel run id-s(ixeper_ids) with run_id-s in IX_experiment headers
% and set this map as internal map for the object
%
ixeper_ids = obj.expdata.get_ixexper_ids();
nruns = numel(ixeper_ids);

unique_runid = unique(ixeper_ids);

id = 1:nruns;
if numel(unique_runid) ~= obj.n_runs || any(isnan(unique_runid))
    ixeper_ids = id;
    exp = obj.expdata_;
    for i=1:nruns
        exp(i).ixexper_id = ixeper_ids(i); % this is a very convoluted way of saying 
                                    % exp(i).run_id = i; but run_ids and id
                                    % are needed to form the map below.
    end
    obj.expdata_ = exp;
    obj.runid_recalculated_ = true;
else
    obj.runid_recalculated_ = false;    
end
obj.ixexperid_map_ = containers.Map(ixeper_ids,id);

