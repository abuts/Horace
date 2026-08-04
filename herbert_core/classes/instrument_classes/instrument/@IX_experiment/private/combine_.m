function [ix_dat_combined,skipped_inputs,this_runid_map,subst_maps] = combine_(exper_cellarray,allow_equal_headers,this_runid_map)
% COMBINE_ : properly combines input IX_experiment cell array with elements
% contained in exper_cellarray, identifying possible duplicates
% and either ignoring them, or throwing error depending on the input
% parameters.
%
% Inputs:
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
% ix_dat_combined -- resulting array, containing unique
%                    instances of IX_experiment classes with
%                    all non-unique IX_experiments excluded.
% skipped_inputs  -- cellarray (with size of input exper_cellarray) of
%                    logical arrays, (each of size of corresponding
%                    exper_cellarray element)containing true where input
%                    object was dropped from output obj and false
%                    where it has been kept.
% this_runid_map --  the map which connects ixdataset_id(s) of data, stored
%                    in the obj with the positions of the data objects in
%                    the object array.
% subst_maps     --  cellarry of new exper_id indices to replace existing
%                    pixel run indices (pointers to appropriate
%                    IX_experiments) if pixels renumbering is necessary. 
%                    if not, empty cellarray.
if isempty(exper_cellarray)
    error('HERBERT:IX_experiment:invalid_argument', ...
        'At least one IX_experiment must be present as firest element of input cellarray')
end
if ~exist("this_runid_map","var")
    this_runid_map = fast_map();
    this_runid_map.trivial_map = true;
end

% Caclulate number of runs defined by all input IX_experiment data
n_runs = cellfun(@(x)numel(x),exper_cellarray);
n_runs = sum(n_runs);

% allocate space for all input headers (final array will be shrinked if not
% all included in the result)
present_runs     = cell(1,n_runs);
pr_hashes        = repmat({''},1,n_runs);

n_experbl_to_add = numel(exper_cellarray);
skipped_inputs = cell(1,n_experbl_to_add);
subst_maps     = cell(1,n_experbl_to_add);
renumerate_pixels= false;
n_found_runs = 0;
for i=1:n_experbl_to_add
    % retrieve arrays for additional IX_experiment-s to add them to result
    add_exper= exper_cellarray{i};
    [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_list,n_subst]=...
        process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,add_exper,allow_equal_headers);
    if n_subst>0
        renumerate_pixels = true;
    end
    subst_maps{i} = subst_list;
    skipped_inputs{i} = skip_runs;
end
if renumerate_pixels
    % if we renumerate pixels, current algorithms requests all pixels id
    % to be updated regardless of them renumerated or not
    for i=1:n_experbl_to_add
        if isempty(subst_maps{i})
            add_exper= exper_cellarray{i};
            n_exper = numel(add_exper);
            subst_list = zeros(2,n_exper);
            for j=1:n_exper
                subst_list(:,j) = add_exper.ixexper_id;
            end
            subst_maps{i} = subst_list;
        end
    end
end

ix_dat_combined = [present_runs{:}];

end
