function [obj,exp_id_array,skipped_inputs,this_runid_map] = combine_(obj,exper_cellarray,allow_equal_headers,varargin)
% COMBINE_ : properly combines input IX_experiment array with elements
% contained in exper_cellarray, identifying possible duplicates
% and either ignoring them, or throwing error depending on the input
% parameters.
%
% Inputs:
% obj             -- single instance or array of IX_experiment objects
% exper_cellarray -- cellarray containing IX_experiment arrays
%                    or Experiment classes to combine their IX_experiments
%                    into obj.
% allow_eq_headers-- if true, headers with the same runid and
%                    same values are allowed and accounted for
%                    in combine operations. If false, routine
%                    throws HORACE:IX_experiment:invalid_argument
%                    if the IX_experiment have the same run_id
%                    and the same values.
% Optional:
% this_runid_map  -- the map containing information about
%                    run_id(s) stored in the object as keys
%                    and pointing to the number of element in obj array
%                    as value.
%
% Returns:
% obj             -- resulting array, containing unique
%                    instances of IX_experiment classes with
%                    all non-unique IX_experiments excluded.
% skipped_inputs  -- cellarray (with size of input exper_cellarray) of
%                    logical arrays, (each of size of corresponding
%                    exper_cellarray element)containing true where input
%                    object was dropped from output obj and false
%                    where it has been kept.
% exp_id_array    -- array contains ix_dataset id-s for each input
%                    IX_experiment value present in exper_cellarray.
%                    Where input IX_experiments with equal ixdataset_num-s
%                    and values are rejected, corresponding
%                    elements of this array contain the
%                    values of rejected ixdataset_num-s. These values will be used
%                    in calculations of pixels run_id for each contributing
%                    file.
% this_runid_map --  the map which connects ixdataset_id(s) of data, stored
%                    in the obj with the positions of the data objects in
%                    the object array.
if nargin<5
    this_runid_map = obj.get_runid_map();
else
    this_runid_map = varargin{1};
end
if isempty(exper_cellarray)
    exp_id_array = arrayfun(@(x)x.ixexper_id,obj);
    obj = arrayfun(@(x)build_hash(x),obj);
    skipped_inputs = {};
    return;
end

if isa(exper_cellarray{1},'Experiment') % extract IX_experiments to combine
    % them with input object.
    exper_cellarray = cellfun(@(x)(x.expdata),exper_cellarray,'UniformOutput',false);
end
n_existing_runs = numel(obj);
% Caclulate number of runs defined by all input IX_experiment data
n_runs = cellfun(@(x)numel(x),exper_cellarray);
n_runs = sum(n_runs)+n_existing_runs;
% Create file_id list for all input headers regardless they are included or
% in final result or not.
exp_id_array = zeros(1,n_runs);
id_now = arrayfun(@(x)x.ixexper_id,obj);
exp_id_array(1:n_existing_runs) = id_now;

% allocate space for all input headers (final array will be shrinked if not
% all included in the result)
present_runs     = cell(1,n_runs);
pr_hashes        = repmat('',1,n_runs);
for i=1:n_existing_runs
    [present_runs{i},pr_hashes{i}] = build_hash(obj(i));
end

n_experbl_to_add = numel(exper_cellarray);
skipped_inputs = cell(1,n_experbl_to_add);
subst_maps     = cell(1,n_experbl_to_add);
renumerate_pixels= false;
for i=1:n_experbl_to_add
    % retrieve arrays for additional IX_experiment-s to add to result
    add_exper= exper_cellarray{i};
    [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map]=...
        process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,add_exper,allow_equal_headers);
    if ~isempty(subst_map)
        renumerate_pixels = true;
    end
    subst_maps{i} = subst_map;
    skipped_inputs{i} = skip_runs;
end
if renumerate_pixels
    for i=1:n_experbl_to_add
        if isempty(subst_maps{i})
        end
    end
end

if numel(obj) ~= n_found_runs
    obj = [present_runs{:}];
end

end
