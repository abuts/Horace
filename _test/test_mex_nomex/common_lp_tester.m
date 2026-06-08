function [t_nomex,t_mex,t_omp] = common_lp_tester( ...
    logo,test_mode, ...
    lp,AB,pix, ...
    n_accum,n_add_out,n_threads,n_points,n_repeats, ...
    sort_pixels,varargin)
% Helper function used in C++ performance tests to evaluate
% performance of different flavour of binning code in wrt MATLAB code
% C++ serial code and C++ OMP code.
%
% Inputs:
% logo      -- text string describing to user what test is running.
% test_mode -- true or false. true used in automated testing, false
%              displays progress information to user and used in manual
%              performance testing. Also, if true, results produced at
%              each iteration by MATLAB, C++ and OMP C++ are compared with
%              each other to ensure equivalence. if false, only final
%              results are compared.
% lp        -- projection class used for transforming pixels
% AB        -- AxesBlock class used for pixel bining.
% pix       -- PixelDataMemory class used as input for processing routine
%              and transformed according to lp and AB above
% n_accum   -- 1 or 3 number of accumulators used in binning defining the
%              test mode. 1 correspond to calculating contribution only and
%              3 -- calculate npix, s and err contributions. 
% n_add_out -- number of additional output variables. Defines what flavour 
%              of binning algorithm should be tested.
% n_threads -- number of OMP threads used by algorithm to test.
% n_repeats -- how many times to repeat the test to collect good statistics
%              and estimate dependence of performance on random input data
%              and OS conditions.
% sort_pixels-- if true, algorithm should also sort resulting pixels 
% varargin   -- optional parameters to pass through to the tested binning
%               algorithms.
% 
% Returns:
% t_nomex  -- array of size n_repeats, containing execution times for the
%             MATLAB code
% t_nomex  -- array of size n_repeats, containing execution times for the
%             cerial C++ code
% t_opm    -- array of size n_repeats, containing execution times of the
%             OMP C++ code.
%

% this will recover existing configuration after test have been
% finished and temporary mex/nomex values will be set within
% the loop.
clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
%
alignment = false;
types_to_sort = {'PixelDataMemory','int64'};
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
acc_nomex = cell(1,in_ac);
acc_mex   = cell(1,in_ac);
acc_omp   = cell(1,in_ac);

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
for i= 1:n_repeats
    if alignment
        pix = pix.set_raw_alignment(al_matr);
    end

    if ~test_mode; fprintf('.'); end
    config_store.instance.set_value('hor_config','use_mex',false);
    inps = copy_inputs(acc_nomex,in_ac,other_inputs,keys);
    t1 = tic();
    [out_nomex{1:n_tout}] = lp.bin_pixels(AB,pix,inps{:});
    t_nomex(i) = toc(t1);
    acc_nomex = copy_accumulators(out_nomex,n_accum);

    config_store.instance.set_value('parallel_config','threads',1);
    config_store.instance.set_value('hor_config','use_mex',true);
    if ~test_mode; fprintf('.'); end
    inps = copy_inputs(acc_mex,in_ac,other_inputs,keys);
    t1 = tic();
    [out_mex{1:n_tout}] = lp.bin_pixels(AB,pix,inps{:});
    t_mex(i) = toc(t1);
    acc_mex = copy_accumulators(out_mex,n_accum);


    config_store.instance.set_value('parallel_config','threads',n_threads);
    if ~test_mode; fprintf('.'); end
    inps = copy_inputs(acc_omp,in_ac,other_inputs,keys);
    t1 = tic();
    [out_omp{1:n_tout}] = lp.bin_pixels(AB,pix,inps{:});
    t_omp(i) = toc(t1);
    if ~test_mode; fprintf('*'); end
    acc_omp = copy_accumulators(out_omp,n_accum);

    if test_mode
        for j=1:n_accum
            assertEqualToTol(out_nomex{j},out_mex{j},'tol',[1.e-9,1.e-9])
            if sort_pixels && ismember(class(acc_omp{i}),types_to_sort)
                nom_data = sort(acc_nomex{j});
                om_data = sort(acc_omp{j});
                assertEqualToTol(nom_data,om_data,'tol',[1.e-9,1.e-9])
            else
                assertEqualToTol(acc_nomex{j},acc_omp{j},'tol',[1.e-9,1.e-9])
            end
        end
    end
    pix.coordinates = rand(4,n_points);
end
if ~test_mode
    disp_perf_results(t_nomex,t_mex,t_omp);
    for j=1:n_accum
        assertEqualToTol(acc_nomex{j},acc_mex{j},'tol',[1.e-9,1.e-9])
        assertEqualToTol(acc_nomex{j},acc_omp{j},'tol',[1.e-9,1.e-9])
    end
    add_acc_pos = 5;
    for j=4:n_tout
        if n_accum>3 && ismember(j,add_acc_pos) % accumulator
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