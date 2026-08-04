function [present_runs,pr_hashes,this_runid_map,n_found_runs,skipped_runs,subst_info,n_subst]=...
    process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,add_exper,allow_equal_headers)
% Add vector of IX_experiment values to vector of existing IX_experiment
% values avoiding adding existing elements.
% Also modify runid_map which maps an IX_experiment element to its position
% in IX_experiment array, so that the mapping remains correct. If initial
% mapping was containing duplicate keys, also return substitusion map which
% would change the duplicated keys to unique keys.
%
% Inputs:
% present_runs    -- cellarray of existing unique IX_experiments added by
%                    previous calls to this routine
% pr_hashes       -- array-helper containing the hashes of unique
%                    experiments above. Created as MATLAB search would not
%                    work on non-standard objects (IX_experiments)
%                    containing these hashes.
% n_found_runs    -- number of runs found during previous calls to this
%                    routine. Contains number of non-empty elements in
%                    present_runs and pr_hashes cellarrays.
% this_runid_map  -- map kontaining pairs:
%                    IX_experients.ixexper_id->num_element
%                    where num_element is the number of IX_experinent
%                    instance in present_runs cellarray.
% add_exper       -- array of IX_experiments to compare against
%                    present_runs, extract elements not in present_runs and
%                    add them to present runs
% allow_equal_headers
%                 -- logical var, which describes behaviour of routine if
%                    the same IX_experiments with different ixexper_id are
%                    present in add_exper array. If it is false, routine
%                    throws error. If true, these IX_experiments are added
%                    as it is.
%
% Returns
% present_runs    -- the cellarray of unique runs modified by adding new unique
%                    runs extracted from add_exper. 
% pr_hashes       -- helper array containing hashes of unique experiments
%                    above, modified by adding hashes of unique
%                    IX_experiments added by this algorithm.
% n_found_runs    -- total number of unique IX_experiments modified 
%                    
% this info
            % will be used to change pixel id(s). Its values are the keys
            % for IX_experiments from different sqw object
n_add_runs   = numel(add_exper);
subst_info   = zeros(2,n_add_runs);
n_subst      = 0;
skipped_runs = false(1,n_add_runs);
for j=1:n_add_runs
    % extract particular IX_experiment to check for addition
    add_IX_exper      = add_exper(j);
    if ~isempty(present_runs{1}) && present_runs{1}.emode ~= add_IX_exper.emode
        error('HORACE:IX_experiment:not_implemented',...
            'you can not currently combine together runs for direct and indirect instruments')
    end

    % hash will be used in comparison below and forever in a future stored
    % in IX_exper. If it is there, it will be just extracted.
    [add_IX_exper,add_hash]  = add_IX_exper.build_hash();
    exper_id                 = add_IX_exper.ixexper_id;


    is_found = ismember(pr_hashes,add_hash);
    duplicate_found = any(is_found);
    if duplicate_found 
        run_there = present_runs{is_found};        
        if ~isempty(run_there.attached_instr_hash) % may be two IX_experiments have different instrument attached.
            % then we must consider them as different
            duplicate_found = isequal(run_there.attached_instr_hash,add_IX_exper.attached_instr_hash);
        end
    end
    if duplicate_found       
        n_subst = n_subst+1;
        subst_info(1,n_subst) = exper_id;
        subst_info(2,n_subst) = add_IX_exper.ixexper_id;
        
        if run_there.ixexper_id == exper_id
            skipped_runs(j) = true;
            continue; % run is already there and taken from duplicated header
            % (combining two cuts from the same sqw object)
        end
        
        % IX_datasets are equal but referred by different pixel id-s
        if ~allow_equal_headers
            i = find(is_found,1);
            existing_ds = present_runs{i};
            error('HORACE:IX_experiment:invalid_argument',[...
                'Can not combine equivalent runs.\n' ...
                'filename, efix, psi, omega, dpsi, gl, gs cannot be the same for two runs with the same run(pixel) identifier\n' ...
                'IX_dataset: N%d, contributed run: %d, filename %s is the same as the already found RunN:%d, Run_id:%d, filename: %s'], ...
                j,add_IX_exper.run_id,add_IX_exper.filename, i,existing_ds.run_id,existing_ds.filename);
        end
        % combine allows duplicated headers and here we add them to the
        % headers and to the map
        n_found_runs     = n_found_runs + 1;
        this_runid_map   = this_runid_map.add(exper_id,n_found_runs);
    else
        % Add new ds with new ID to the combined ds
        n_found_runs     = n_found_runs + 1;

        if this_runid_map.isKey(exper_id) % two different IX_datasets are
            % referred by the same id. (combining two independent sqw objects
            % built separately) Pixel id-s must be updated.
            add_IX_exper.ixexper_id = n_found_runs;
            n_subst = n_subst+1;
            subst_info(1,n_subst) = exper_id;
            subst_info(2,n_subst) = add_IX_exper.ixexper_id;
            exper_id  = add_IX_exper.ixexper_id;
        end
        this_runid_map   = this_runid_map.add(exper_id,n_found_runs);

    end
    present_runs{n_found_runs}= add_IX_exper;
    pr_hashes{n_found_runs}   = add_hash;
end % endfor
end
