function [t_mex,t_nomex,t_omp] = common_ab_tester( ...
    logo,test_mode,AB,n_accum,n_add_in,n_add_out,n_threads,n_points,n_repeats, ...
    sort_pixels,varargin)
% this will recover existing configuration after test have been
% finished and temporary mex/nomex values will be set within
% the loop.

clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
%
cell_data = iscell(n_add_in);
if cell_data
    n_add_in = n_add_in{1};
end
types_to_sort = {'PixelDataMemory','int64'};
alignment = false;
keys = {};
if numel(varargin)>0
    if isnumeric(varargin{1})
        alignment = true;
        al_matr = varargin{1};
    else
        keys{1} = varargin{1};
    end
end

acc_nomex = cell(1,n_accum);
acc_mex = cell(1,n_accum);
acc_omp = cell(1,n_accum);

n_tout = n_accum+n_add_out;
out_nomex = cell(1,n_tout);
out_mex = cell(1,n_tout);
out_omp = cell(1,n_tout);

t_nomex = zeros(1,n_repeats);
t_mex  = zeros(1,n_repeats);
t_omp  = zeros(1,n_repeats);
other_inputs = {};
if ~test_mode
    disp(logo)
end

pix_data = rand(9,n_points);
pix_id = 500+floor(100*rand(1,n_points));
det_ids = 1024+floor(100000*rand(1,n_points));
en_ids = floor(500*rand(1,n_points));
pix_data(PixelDataBase.field_index('run_idx'),:)     = pix_id;
pix_data(PixelDataBase.field_index('detector_idx'),:) = det_ids;
pix_data(PixelDataBase.field_index('energy_idx'),:)   = en_ids;
pix = PixelDataMemory(pix_data);

for i= 1:n_repeats
    if n_add_in>0 && ~cell_data
        pix.run_idx = 500+floor(100*rand(1,n_points));
        pix = PixelDataMemory(pix_data);
        in_coord = pix.coordinates;
        if alignment
            pix = pix.set_raw_alignment(al_matr);
        end
        other_inputs{1} = pix;
    else
        in_coord = rand(4,n_points);
        if cell_data
            sig = rand(1,n_points);
            err = rand(1,n_points);
            other_inputs{1} = {sig,err};
        end
    end
    if ~test_mode; fprintf('.'); end
    config_store.instance.set_value('hor_config','use_mex',false);
    inps = copy_inputs(acc_nomex,n_accum,other_inputs,keys);
    t1 = tic();
    [out_nomex{1:n_tout}] = AB.bin_pixels(in_coord,inps{:});
    t_nomex(i) = toc(t1);
    acc_nomex = copy_accumulators(out_nomex,n_accum);

    config_store.instance.set_value('parallel_config','threads',1);
    config_store.instance.set_value('hor_config','use_mex',true);
    if ~test_mode; fprintf('.'); end
    inps = copy_inputs(acc_mex,n_accum,other_inputs,keys);
    t1 = tic();
    [out_mex{1:n_tout}] = AB.bin_pixels(in_coord,inps{:});
    t_mex(i) = toc(t1);
    acc_mex = copy_accumulators(out_mex,n_accum);


    config_store.instance.set_value('parallel_config','threads',n_threads);
    if ~test_mode; fprintf('.'); end
    inps = copy_inputs(acc_omp,n_accum,other_inputs,keys);
    t1 = tic();
    [out_omp{1:n_tout}] = AB.bin_pixels(in_coord,inps{:});
    t_omp(i) = toc(t1);
    if ~test_mode; fprintf('*'); end
    acc_omp = copy_accumulators(out_omp,n_accum);

    if test_mode
        for j=1:n_tout
            assertEqualToTol(out_nomex{j},out_mex{j},'tol',[1.e-9,1.e-9])
            if sort_pixels && ismember(class(out_omp{j}),types_to_sort)
                nom_data = sort(out_nomex{j});
                om_data = sort(out_omp{j});
                assertEqualToTol(nom_data,om_data,'tol',[1.e-9,1.e-9])
            else
                assertEqualToTol(out_nomex{j},out_omp{j},'tol',[1.e-9,1.e-9])
            end
        end
    end
end
if ~test_mode
    disp_perf_results(t_nomex,t_mex,t_omp);
    for j=1:n_accum
        assertEqualToTol(acc_nomex{j},acc_mex{j},'tol',[1.e-9,1.e-9])
        assertEqualToTol(acc_nomex{j},acc_omp{j},'tol',[1.e-9,1.e-9])
    end
    for j=4:n_tout
        if n_accum>3 && j==5 % accumulator
            continue;
        end

        assertEqualToTol(out_nomex{j},out_mex{j},'tol',[1.e-9,1.e-9])
        if sort_pixels && ismember(class(out_omp{j}),types_to_sort)
            nom_data = sort(out_nomex{j});
            om_data = sort(out_omp{j});
            assertEqualToTol(nom_data,om_data,'tol',[1.e-9,1.e-9])
        else
            assertEqualToTol(out_nomex{j},out_omp{j},'tol',[1.e-9,1.e-9])
        end
    end
end
end


function inps = copy_inputs(accum,n_accum,other_inputs,keys)
n_other = numel(other_inputs);
n_keys = 0;
if nargin >3
    n_keys = numel(keys);
end
n_tot = n_accum+n_other+n_keys ;
inps = cell(1,n_tot);

for i=1:n_accum
    inps{i} = accum{i};
end
for i=1:n_other
    inps{i+n_accum} = other_inputs{i};
end
for i=1:n_keys
    inps{i+n_accum+n_other} = keys{i};
end
if n_accum>3
    acc4 = inps{4};
    inps{4}=inps{n_accum+1};
    inps{n_accum+1} = acc4;
end

end

function acc = copy_accumulators(outs,n_accum)
acc = cell(1,n_accum);
for i=1:n_accum
    if i>3
        acc{i} = outs{i+1};
    else
        acc{i} = outs{i};
    end
end
end