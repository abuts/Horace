function run_files = set_up_unique_exper_id(run_files)
% Processes the list of runfiles and assign 
%
% Input:
% run_files  - cellarray of rundata class instances with some run_id
%              may be duplicated or some cells are empty.
% Output:
% run_files  - cellarray of the same rundata class instances. If some
%              original id-s are missing or duplicated, have exper_id
%              property set to the number of the object in the list.
%              If run_id are defined properly, all id-s are set.

% extract exisitng run_id(s) assigning NaN to empty cells
run_ids_all = cellfun(@get_run_id,run_files,'UniformOutput',true);

unique_id = unique(run_ids_all);
duplicated_id = numel(unique_id) ~= numel(run_ids_all);
missing_id = any(isnan(run_ids_all));
if duplicated_id || missing_id
    for i=1:numel(run_files)
        if isa(run_files{i},'rundata')
            run_files{i}.exper_id = i;
        end
    end
end
end

function id = get_run_id(run)
if isempty(run)
    id = NaN;
else
    id = run.run_id;
end
end