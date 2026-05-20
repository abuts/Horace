function t_omp = threads_impact_tester( ...
    logo,test_mode, ...
    lp,AB,pix, ...
    n_accum,n_add_out,n_threads,n_repeats, ...
    varargin)
% this will recover existing configuration after test have been
% finished and temporary mex/nomex values will be set within
% the loop.


%
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
in_ac = n_accum;
if in_ac == 1
    in_ac = 3;
end
acc_omp   = cell(1,in_ac);

n_tout = n_accum+n_add_out;
out_omp = cell(1,n_tout);

t_omp  = zeros(1,n_repeats);
other_inputs = {};
if ~test_mode
    disp(logo)
end
config_store.instance.set_value('parallel_config','threads',n_threads);
config_store.instance.set_value('hor_config','use_mex',true);
for i= 1:n_repeats
    if alignment
        pix = pix.set_raw_alignment(al_matr);
    end

    if ~test_mode; fprintf('.'); end
    inps = copy_inputs(acc_omp,in_ac,other_inputs,keys);
    t1 = tic();
    [out_omp{1:n_tout}] = lp.bin_pixels(AB,pix,inps{:});
    t_omp(i) = toc(t1);
    if ~test_mode; fprintf('*'); end
    acc_omp = copy_accumulators(out_omp,n_accum);
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

n_acc_provided = numel(accum);
for i=1:n_accum
    if i>n_acc_provided
        inps{i} = [];
    else
        inps{i} = accum{i};
    end
end
for i=1:n_other
    inps{i+n_accum} = other_inputs{i};
end
for i=1:n_keys
    inps{i+n_accum+n_other} = keys{i};
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