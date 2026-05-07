classdef performance_bin_pixels_mex_nomex
    % Series of tests to check work of mex files against Matlab files

    properties
        this_folder;
        no_mex;
        perf_tests_list
        result_name
        perf_results;
    end

    methods
        function obj=performance_bin_pixels_mex_nomex(varargin)

            obj.this_folder = fileparts(which('performance_bin_pixels_mex_nomex.m'));

            [~,n_errors] = check_horace_mex();
            obj.no_mex = n_errors > 0;
            test_names = {...
                'mode7p (proj pixels + unique runid + return idx + selected)',... 7
                'mode6p (proj pixels + unique runid + return idx)',...            6
                'mode5p (proj and sort pixels + unique runid)',...                5
                'mode4p (proj and sort pixels applying alignment)',...            4a
                'mode4p (proj and sort pixels)',...                               4
                'mode2p (proj pixels + sig_err)',...                              2
                'mode0p (proj coord, calc npix)'...                               0
                'mode7 (bin pixels + unique runid + return idx + selected)',... 7
                'mode6 (bin pixels + unique runid + return idx)',...            6
                'mode5 (bin and sort pixels + unique runid)',...                5
                'mode4 (bin and sort pixels applying alignment)',...            4a
                'mode4 (profile bin and sort pixels)',...                       4p
                'mode4 (bin and sort pixels)',...                               4
                'mode3 (binning cellarrays of data over coordinate frame)',...  3
                'mode2 (bin pixels + sig_err)',...                              2
                'mode0 (bin coord, calc npix)'...
                };
            test_func = {...
                @()performance_proj_mex_nomex_mode7_nosort_idx_sel(obj),... 7
                @()performance_proj_mex_nomex_mode6_nosort_idx(obj),...     6
                @()performance_proj_mex_nomex_mode5_sort_and_uid(obj),...   5
                @()performance_proj_mex_nomex_mode4_and_align(obj),...      4a
                @()performance_proj_mex_nomex_mode4(obj),...                4
                @()performance_proj_mex_nomex_mode2_npix_and_sigerr(obj),...2
                @()performance_proj_mex_nomex_mode0(obj)...                 0
                @()performance_mex_nomex_mode7_nosort_idx_sel(obj),... 7
                @()performance_mex_nomex_mode6_nosort_idx(obj),...     6
                @()performance_mex_nomex_mode5_sort_and_uid(obj),...   5
                @()performance_mex_nomex_mode4_and_align(obj),...      4a
                @()performance_mex_mode4_for_profile_sort_pix(obj),... 4p
                @()performance_mex_nomex_mode4(obj),...                4
                @()performance_mex_nomex_mode3_sigerr_cell(obj),...    3
                @()performance_mex_nomex_mode2_npix_and_sigerr(obj),...2
                @()performance_mex_nomex_mode0(obj)...
                };

            obj.perf_tests_list = containers.Map(test_names,test_func);
            obj.result_name = [getComputerName(),'_mex_performance_',char(datetime('today'))];
        end
        %==================================================================
        function run_all_performance_tests(obj)
            meth_names = obj.perf_tests_list.keys();

            perf_res = struct();
            for i=1:numel(meth_names)
                mode_name = split(meth_names{i});
                fh = obj.perf_tests_list(meth_names{i});
                [t_nomex,t_mex] = fh();
                mode_name = mode_name{1};
                perf_res.(mode_name) = [t_nomex(:)';t_mex(:)'];
                fprintf('\n');
            end
            obj.perf_results = perf_res;
            if isfile(obj.result_name)
                [fp,fn,fe] = fileparts(obj.result_name);
                back_file_name = fullfile(fp,[fn,'_bak',fe]);
                movefile(obj.result_name,back_file_name,'f');
            end
            save(obj.result_name,'perf_res');
        end
        %==================================================================
        %==================================================================
        function [t_nomex,t_mex] = performance_proj_mex_nomex_mode7_nosort_idx_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** P performance mode7 (bin pixels + unique runid + return idx + selected):")
            for i= 1:n_repeats
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom,is_sel_nom] = ...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);

                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex,is_sel_mex] =...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);
                t_mex(i) = toc(t1);

                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(int64(pix_idx_nom),pix_idx_mex);
                assertEqual(is_sel_nom,is_sel_mex);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
                pix.run_idx  =  500+floor(100*rand(1,n_points));

            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE DATA: ndw2671
        end

        function [t_nomex,t_mex] = performance_mex_nomex_mode7_nosort_idx_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,1,50,20]);

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Mex/nomex performance mode7 (bin pixels + unique runid + return idx + selected):")
            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                ids = 500+floor(100*rand(1,n_points));
                pix_data(PixelDataBase.field_index('run_idx'),:) = ids;

                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;
                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom,is_sel_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex,is_sel_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix,uniqId_mex);
                t_mex(i) = toc(t1);

                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(int64(pix_idx_nom),pix_idx_mex);
                assertEqual(is_sel_nom,is_sel_mex);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE DATA: ndw2671
            %*** Mex/nomex performance mode7 (bin pixels + unique runid + return idx + selected):
            %*** time of first step,    nomex:  3.1(sec)  mex:  1.2(sec); Acceleration :  2.5
            %*** Average time per step, nomex:  3.1(sec)  mex:  1.3(sec); Acceleration :  2.3
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex] = performance_proj_mex_nomex_mode6_nosort_idx(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Proj mex/nomex performance mode6 (bin pixels + unique runid + return idx):")
            for i= 1:n_repeats
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom] =...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);
                t_mex(i) = toc(t1);

                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(int64(pix_idx_nom),pix_idx_mex);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
                pix.run_idx  =  500+floor(100*rand(1,n_points));
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE DATA: ndw2671
        end
        function [t_nomex,t_mex] = performance_mex_nomex_mode6_nosort_idx(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);

            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Mex/nomex performance mode6 (bin pixels + unique runid + return idx):")
            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                ids = 500+floor(100*rand(1,n_points));
                pix_data(PixelDataBase.field_index('run_idx'),:) = ids;

                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;
                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix,uniqId_mex);
                t_mex(i) = toc(t1);

                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(int64(pix_idx_nom),pix_idx_mex);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE DATA: ndw2671
            %*** Mex/nomex performance mode6 (bin pixels + unique runid + return idx):
            %*** time of first step,    nomex:  3.8(sec)  mex:  1.3(sec); Acceleration :  2.6-3
            %*** Average time per step, nomex:  3.3(sec)  mex:  1.4(sec); Acceleration :  2.4
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex] = performance_proj_mex_nomex_mode5_sort_and_uid(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Proj mex/nomex performance mode5 (bin and sort pixels + unique runid):")
            for i= 1:n_repeats
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] =...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);
                %AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);

                t_mex(i) = toc(t1);

                assertEqual(uint32(uniqId_nom),uniqId_mex);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
                pix.run_idx  =  500+floor(100*rand(1,n_points));
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFEFENCE Data NDW2671
            %*** Mex/nomex performance mode5 (bin and sort pixels + unique runid):
            %*** time of first step,    nomex:  5.8(sec)  mex:  2.1(sec); Acceleration :  2.7
            %*** Average time per step, nomex:  5.7(sec)  mex:  2.1(sec); Acceleration :  2.7
        end
        function [t_nomex,t_mex] = performance_mex_nomex_mode5_sort_and_uid(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Mex/nomex performance mode5 (bin and sort pixels + unique runid):")
            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                ids = 500+floor(100*rand(1,n_points));
                pix_data(PixelDataBase.field_index('run_idx'),:) = ids;

                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;
                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix,uniqId_mex);
                t_mex(i) = toc(t1);

                assertEqual(uint32(uniqId_nom),uniqId_mex);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFEFENCE Data NDW2671
            %*** Mex/nomex performance mode5 (bin and sort pixels + unique runid):
            %*** time of first step,    nomex:  5.8(sec)  mex:  2.1(sec); Acceleration :  2.7
            %*** Average time per step, nomex:  5.7(sec)  mex:  2.1(sec); Acceleration :  2.7
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex] = performance_mex_mode4_for_profile_sort_pix(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', true,'log_level',-1);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 5;
            npix_mex   = []; s_mex   = [];  e_mex  =[];
            npix_nomex = []; s_nomex = [];  e_nomex=[];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            pix = PixelDataMemory();
            disp("*** Mex performance mode4 (profile bin and sort pixels):")
            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                pix = pix.set_raw_data(pix_data);
                coord = pix.coordinates;

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix);
                t_mex(i) = toc(t1);


            end
            obj.disp_perf_results(t_nomex,t_mex);
            % tav_mex = sum(t_mex)/n_repeats;
            % fprintf( ...
            %     '\n*** time of first step : %4.2g(sec)  av time per step: %4.2g(sec)\n', ...
            %     t_mex(1),tav_mex);
            % REFERENCE Data ndw2671
            %*** Mex performance mode4 (bin and sort pixels):
            %*** time of first step :  1.9(sec)  av time per step:  1.8(sec)
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex] = performance_proj_mex_nomex_mode4_and_align(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];

            al_matr = rotvec_to_rotmat([10,20,15]);

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Proj mex/nomex performance mode4 (bin and sort pixels applying alignment):")
            for i= 1:n_repeats
                fprintf('.')

                % set alignment matrix but do not apply alignment.
                % (Simulate filebased pixels)
                pix.alignment_matr = al_matr;

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = ...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();

                [npix_mex,s_mex,e_mex,pix_ok_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex);

                t_mex(i) = toc(t1);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE Data ndw2671
        end
        function [t_nomex,t_mex] = performance_mex_nomex_mode4_and_align(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];

            al_matr = rotvec_to_rotmat([10,20,15]);

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Mex/nomex performance mode4 (bin and sort pixels applying alignment):")
            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                pix = PixelDataMemAlTester(pix_data);
                coord = pix.coordinates;
                % set alignment matrix but do not apply alignment.
                % (Simulate filebased pixels)
                pix.alignment_matr = al_matr;

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix);
                t_mex(i) = toc(t1);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode4 (bin and sort pixels applying alignment):
            %*** time of first step,    nomex:  5.9(sec)  mex:  1.7(sec); Acceleration :  3.4
            %*** Average time per step, nomex:  6.1(sec)  mex:  1.9(sec); Acceleration :  3.2
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex] = performance_proj_mex_nomex_mode4(obj)

            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end

            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);

            n_repeats = 5;

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();

            npix_nomex=[]; s_nomex=[]; e_nomex=[];
            npix_mex=[];   s_mex=[];   e_mex=[];

            disp("*** Proj mex/nomex performance mode4 (sort pixels):")
            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            for i=1:n_repeats
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = ...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex);
                t_nomex(i) = toc(t1);
                fprintf('.')

                t1 = tic();
                config_store.instance.set_value('hor_config','use_mex',true);
                [npix_mex,s_mex,e_mex,pix_ok_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex);
                t_mex(i) = toc(t1);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1e-12 1e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1e-12 1e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1e-12 1e-12])

                pix.coordinates = rand(4,n_points);
            end

            obj.disp_perf_results(t_nomex,t_mex);
        end
        function [t_nomex,t_mex] = performance_mex_nomex_mode4(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Mex/nomex performance mode4 (bin and sort pixels):")
            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix);
                t_mex(i) = toc(t1);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode4 (bin and sort pixels):
            %*** time of first step,    nomex:  5.4(sec)  mex:  1.7(sec); Acceleration :  3.1
            %*** Average time per step, nomex:  5.4(sec)  mex:  1.8(sec); Acceleration :    3
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex] = performance_mex_nomex_mode3_sigerr_cell(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            %
            [AB,n_points]=obj.prepare_clean_bin_data();

            n_repeats = 10;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Mex/nomex performance mode3 (binning cellarrays of data over coordinate frame):")
            for i= 1:n_repeats
                fprintf('.')
                coord = rand(4,n_points);
                sig = rand(1,n_points);
                err = rand(1,n_points);

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,{sig,err});
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,{sig,err});
                t_mex(i) = toc(t1);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-9,1.e-9])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-9,1.e-9])
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode3 (binning cellarrays of data over coordinate frame):
            %*** time of first step,    nomex:  2.2(sec)  mex:  0.7(sec); Acceleration :  3.2
            %*** Average time per step, nomex:  2.2(sec)  mex: 0.69(sec); Acceleration :  3.3
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex] = performance_proj_mex_nomex_mode2_npix_and_sigerr(obj)

            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end

            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);

            n_repeats = 10;

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();

            npix_nomex = []; s_nomex = []; e_nomex=[];
            npix_mex   = []; s_mex   = []; e_mex=[];

            disp("*** Proj mex/nomex performance mode2 (bin pixels + sig_err):")
            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            for i=1:n_repeats
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',false);
                t1=tic();
                [npix_nomex,s_nomex,e_nomex] = lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex);
                t_nomex(i)=toc(t1);

                fprintf('.')
                config_store.instance.set_value('hor_config','use_mex',true);
                t1=tic();
                [npix_mex,s_mex,e_mex] = lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex);
                t_mex(i)=toc(t1);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1e-9 1e-9])
                assertEqualToTol(e_nomex,e_mex,'tol',[1e-9 1e-9])

                pix.coordinates = rand(4,n_points);
            end

            obj.disp_perf_results(t_nomex,t_mex);
        end

        function [t_nomex,t_mex] = performance_mex_nomex_mode2_npix_and_sigerr(obj)

            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            %
            [AB,n_points]=obj.prepare_clean_bin_data();

            n_repeats = 10;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Mex/nomex performance mode2 (bin pixels + sig_err):")
            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                [npix_mex,s_mex,e_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix);
                t_mex(i) = toc(t1);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-9,1.e-9])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-9,1.e-9])
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode2 (bin pixelsbin pixels + sig_err):
            %*** time of first step,    nomex:  3.4(sec)  mex: 0.68(sec); Acceleration :    5
            %*** Average time per step, nomex:  3.4(sec)  mex: 0.71(sec); Acceleration :  4.8
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex] = performance_proj_mex_nomex_mode0(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            %

            n_repeats = 10;

            [lp,AB,pix,n_points]=obj.prepare_lp_bin_data();

            npix_nomex = [];
            npix_mex   = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Proj mex/nomex performance mode0 (bin coord, calc npix):")
            for i= 1:n_repeats
                fprintf('.')
                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                npix_nomex  = lp.bin_pixels(AB,pix,npix_nomex,[],[]);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                npix_mex = lp.bin_pixels(AB,pix,npix_mex,[],[]);
                t_mex(i) = toc(t1);
                pix.coordinates = rand(4,n_points);
                assertEqual(npix_nomex,npix_mex)
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE Data ndw2671
        end

        function [t_nomex,t_mex] = performance_mex_nomex_mode0(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            %
            [AB,n_points]=obj.prepare_clean_bin_data();

            n_repeats = 10;
            npix_nomex = [];
            npix_mex   = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            disp("*** Mex/nomex performance mode0 (bin coord, calc npix):")
            for i= 1:n_repeats
                fprintf('.')
                in_coord = rand(4,n_points);
                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                npix_nomex = AB.bin_pixels(in_coord,npix_nomex);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);

                t1 = tic();
                npix_mex = AB.bin_pixels(in_coord,npix_mex);
                t_mex(i) = toc(t1);

                assertEqual(npix_nomex,npix_mex)
            end
            obj.disp_perf_results(t_nomex,t_mex);
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode0 (bin coord, calc npix):
            %*** time of first step,    nomex:  1.4(sec)  mex: 0.61(sec); Acceleration :  2.3
            %*** Average time per step, nomex:  1.4(sec)  mex: 0.62(sec); Acceleration :  2.2
        end

    end
    methods(Static)
        function [AB,n_points]=prepare_clean_bin_data(nbins_all_dims)
            if nargin == 0
                nbins_all_dims = [50,1,50,1];
            end
            AB = AxesBlockBase_tester('nbins_all_dims',nbins_all_dims, ...
                'img_range',[0,0,0,0;1,0.8,1,0.8]);
            n_points = 200;
        end

        function [lp,ab,pix,n_points]=prepare_lp_bin_data()
            ab = line_axes('nbins_all_dims',[50,1,50,1], ...
                'img_range',[0,0,0,0;1,0.8,1,0.8]);
            n_points = 20000000;
            pix_id = 10;
            pix_coord = rand(9,n_points);
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;
            pix_coord(PixelDataBase.field_index('run_idx'),10:20) = 2*pix_id;

            pix = PixelDataMemory(pix_coord);
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[0,0,0]);
        end

        function [tav_nom,tav_mex] = disp_perf_results(t_nomex,t_mex)
            n_repeats = numel(t_nomex);
            tav_mex = sum(t_mex)/n_repeats;
            tav_nom = sum(t_nomex)/n_repeats;
            fprintf( ...
                '\n*** time of first step,    nomex: %4.2g(sec)  mex: %4.2g(sec); Acceleration : %4.2g\n', ...
                t_nomex(1),t_mex(1),t_nomex(1)/t_mex(1));
            fprintf( ...
                '*** Average time per step, nomex: %4.2g(sec)  mex: %4.2g(sec); Acceleration : %4.2g\n', ...
                tav_nom,tav_mex,tav_nom/tav_mex);
        end
    end
end
