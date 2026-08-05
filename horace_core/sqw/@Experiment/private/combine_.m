function [exper_comb,nspe,pix_id_subst] = combine_(exp_cellarray,allow_equal_headers)
%COMBINE_
% Take cellarray of experiments (e.g., generated from each runfile build
% during gen_sqw generation)
% and combine then together into single Experiment info class
% Inputs:
% obj           -- first experiment to add other experiments to.
% exp_cellarray -- additional Experiment class or cellarray of Experiment
%                  classes, related to different runs or combination of runs
%
% allow_equal_headers
%               -- if true, equal runs are allowed.
%                At present, we insist that the contributing spe data are distinct
%                in that:
%                - filename, run_id,efix, psi, omega, dpsi, gl, gs cannot
%                  all be equal for two spe data input. If
%                  allow_equal_headers is set to true, this check is
%                  disabled
%
% Returns:
% exper_comb    -- Experiment class containing combined input
%                  experiments.
% nspe          -- number of unique runs, contributing into
%                  resulting Experiment
% pix_id_subst  -- cellarray of arrays of final run_id-s for all input nxspe.
%                  if not empty, pixels ixexper_ids have to be renumerated
%                  accorting to this table as original table was containing
%                  the same pixel-id(s) pointing to different IX_experiment
%                  data.

if isempty(exp_cellarray)|| numel(exp_cellarray)== 0
    exper_comb = [];
    nspe = 0;
    pix_id_subst = {};
    return;
end
if isa(exp_cellarray,'Experiment')
    if isscalar(exp_cellarray)
        exp_cellarray = {exp_cellarray};
    else
        exp_cellarray = num2cell(exp_cellarray);
    end
end

n_contrib = numel(exp_cellarray);
nspe      = zeros(n_contrib,1);
exp_info  = cell(1,n_contrib);
% extract intrument/sample/detector information to check different
% experiments have same/different enviroment for proper comparison.
for i=1:n_contrib
    nspe(i)   = exp_cellarray{i}.n_runs;
    exp_info{i} = exp_cellarray{i}.expdata;

    instr  = exp_cellarray{i}.instruments;
    sample = exp_cellarray{i}.samples;
    det_arr= exp_cellarray{i}.detector_arrays;
    exp_data = exp_info{i};
    for j=1:nspe(i)
        [instr(j),inst_hash] = build_hash(instr(j));
        [det_arr(j),det_hash] = build_hash(det_arr(j));
        exp_data(j).attached_instr_hash = [inst_hash,det_hash];
    end
    exp_cellarray{i}.instruments     = instr;
    exp_cellarray{i}.samples         = sample;
    exp_cellarray{i}.detector_arrays = det_arr;
    exp_info{i}                      = exp_data;
end

[expinfo,skipped_runs,~,pix_id_subst]    = IX_experiment.combine(exp_info,allow_equal_headers);

instr   = unique_references_container('IX_inst');
sampl   = unique_references_container('IX_samp');
det     = unique_references_container('IX_detector_array');
ic = 0;
for i=1:n_contrib
    skipped_run = skipped_runs{i};
    for j=1:exp_cellarray{i}.n_runs
        if skipped_run(j) % the run have been rejected
            continue;
        end
        ic = ic+1;
        instr{ic}  = exp_cellarray{i}.instruments{j};
        sampl{ic}  = exp_cellarray{i}.samples{j};
        det{ic}    = exp_cellarray{i}.detector_arrays{j};
        % Check experiment consistency:
        if sampl{ic} ~= sampl{1}
            error('HORACE:Experiment:runtime_error',[...
                'The sample for all runs contributing to experiment have to be the same.\n',...
                'File N%d, contributed run %d differs from the first run '],i,j);
        end

    end
end
nspe = ic; % number of contributing experiments has been counted here
exper_comb = Experiment(det,instr,sampl,expinfo);
