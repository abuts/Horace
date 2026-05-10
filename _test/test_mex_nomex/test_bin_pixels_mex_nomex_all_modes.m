classdef test_bin_pixels_mex_nomex_all_modes < TestCase
    % Series of tests to check work of mex files against Matlab files

    properties
        this_folder;
        no_mex;
        n_threads;
    end

    methods
        function obj=test_bin_pixels_mex_nomex_all_modes(varargin)
            if nargin>0
                name=varargin{1};
            else
                name = 'test_bin_pixels_mex_nomex_all_modes';
            end
            obj = obj@TestCase(name);

            obj.n_threads = 8;
            [~,n_errors] = check_horace_mex();
            obj.no_mex = n_errors > 0;
        end
        %==================================================================
        %==================================================================
        function test_proj_mex_nomex_mode8_nosort_idx_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            npix_omp   = []; s_omp = [];  e_omp=[];uniqId_omp = [];            

            for i= 1:n_repeats

                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom,is_sel_nom] = ...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex,is_sel_mex] =...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);


                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp,uniqId_omp,pix_idx_omp,is_sel_omp] = ...
                    lp.bin_pixels(AB,pix,npix_omp,s_omp,e_omp,uniqId_omp);



                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(uint32(uniqId_nom),uniqId_omp);
                assertEqual(int64(pix_idx_nom),pix_idx_mex);
                assertEqual(int64(pix_idx_nom),pix_idx_omp);
                assertEqual(is_sel_nom,is_sel_mex);
                assertEqual(is_sel_nom,is_sel_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
                pix.run_idx  =  500+floor(100*rand(1,n_points));

            end
        end

        function test_mex_nomex_mode8_nosort_idx_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,1,50,20]);

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            npix_omp   = []; s_omp = [];  e_omp=[];uniqId_omp = [];

            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                ids = 500+floor(100*rand(1,n_points));
                pix_data(PixelDataBase.field_index('run_idx'),:) = ids;

                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;
                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom,is_sel_nom] = ...
                    AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex,is_sel_mex] =...
                    AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix,uniqId_mex);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp,uniqId_omp,pix_idx_omp,is_sel_omp] = ...
                    AB.bin_pixels(coord,npix_omp,s_omp,e_omp,pix,uniqId_omp);


                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(uint32(uniqId_nom),uniqId_omp);
                assertEqual(int64(pix_idx_nom),pix_idx_mex);
                assertEqual(int64(pix_idx_nom),pix_idx_omp);
                assertEqual(is_sel_nom,is_sel_mex);
                assertEqual(is_sel_nom,is_sel_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])
            end
        end
        %------------------------------------------------------------------
        function test_proj_mex_nomex_mode7_nosort_id_idx(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            npix_omp   = []; s_omp = [];  e_omp=[];uniqId_omp = [];

            for i= 1:n_repeats

                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom] =...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp,uniqId_omp,pix_idx_omp] = ...
                    lp.bin_pixels(AB,pix,npix_omp,s_omp,e_omp,uniqId_omp);

                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(uint32(uniqId_nom),uniqId_omp);
                assertEqual(int64(pix_idx_nom),pix_idx_mex);
                assertEqual(int64(pix_idx_nom),pix_idx_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
                pix.run_idx  =  500+floor(100*rand(1,n_points));
            end
        end
        function test_mex_nomex_mode7_nosort_id_idx(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);

            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            npix_omp   = []; s_omp = [];  e_omp=[];uniqId_omp = [];

            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                ids = 500+floor(100*rand(1,n_points));
                pix_data(PixelDataBase.field_index('run_idx'),:) = ids;

                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;
                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom] = ...
                    AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex] =...
                    AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix,uniqId_mex);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp,uniqId_omp,pix_idx_omp] = ...
                    AB.bin_pixels(coord,npix_omp,s_omp,e_omp,pix,uniqId_omp);

                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(uint32(uniqId_nom),uniqId_omp);
                assertEqual(int64(pix_idx_nom),pix_idx_mex);
                assertEqual(int64(pix_idx_nom),pix_idx_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])
            end
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function test_mex_nomex_mode6_nosort_id(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);

            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            npix_omp   = []; s_omp = [];  e_omp=[];uniqId_omp = [];

            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                ids = 500+floor(100*rand(1,n_points));
                pix_data(PixelDataBase.field_index('run_idx'),:) = ids;

                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;
                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] = ...
                    AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom,'-nosort');

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] =...
                    AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix,uniqId_mex,'-nosort');


                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp,uniqId_omp] = ...
                    AB.bin_pixels(coord,npix_omp,s_omp,e_omp,pix,uniqId_omp,'-nosort');


                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(uint32(uniqId_nom),uniqId_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])
            end
        end
        %------------------------------------------------------------------
        function test_proj_mex_nomex_mode5_sort_and_uid(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            npix_omp   = []; s_omp = [];  e_omp=[];uniqId_omp = [];

            for i= 1:n_repeats
                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] =...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);
                %AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp,uniqId_omp] = ...
                    lp.bin_pixels(AB,pix,npix_omp,s_omp,e_omp,uniqId_omp);

                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(uint32(uniqId_nom),uniqId_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_omp,npix_omp)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(s_nomex,s_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
                pix.run_idx  =  500+floor(100*rand(1,n_points));
            end
        end

        function test_mex_nomex_mode5_sort_and_uid(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            npix_omp   = []; s_omp = [];  e_omp=[];uniqId_omp = [];
            for i= 1:n_repeats

                pix_data = rand(9,n_points);
                ids = 500+floor(100*rand(1,n_points));
                pix_data(PixelDataBase.field_index('run_idx'),:) = ids;

                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;
                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix,uniqId_nom);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix,uniqId_mex);


                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp,uniqId_omp] = AB.bin_pixels(coord,npix_omp,s_omp,e_omp,pix,uniqId_omp);


                assertEqual(uint32(uniqId_nom),uniqId_mex);
                assertEqual(uint32(uniqId_nom),uniqId_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_omp,npix_omp)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(s_nomex,s_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])
            end
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function test_proj_mex_nomex_mode4_and_align(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];
            npix_omp   = []; s_omp = [];  e_omp=[];

            al_matr = rotvec_to_rotmat([10,20,15]);

            for i= 1:n_repeats
                % set alignment matrix but do not apply alignment.
                % (Simulate filebased pixels)
                pix = pix.set_raw_alignment(al_matr);

                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = ...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp] = ...
                    lp.bin_pixels(AB,pix,npix_omp,s_omp,e_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_nomex,npix_omp)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(s_nomex,s_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
            end
        end
        function test_mex_nomex_mode4_and_align(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];

            al_matr = rotvec_to_rotmat([10,20,15]);

            for i= 1:n_repeats
                fprintf('.')
                pix_data = rand(9,n_points);
                pix = PixelDataMemAlTester(pix_data);
                coord = pix.coordinates;
                % set alignment matrix but do not apply alignment.
                % (Simulate filebased pixels)
                pix = pix.set_raw_alignment(al_matr);


                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix);

                config_store.instance.set_value('hor_config','use_mex',true);
                [npix_mex,s_mex,e_mex,pix_ok_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
            end
        end
        %------------------------------------------------------------------
        function test_proj_mex_nomex_mode4(obj)

            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end

            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);

            n_repeats = 2;

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();

            npix_nomex=[]; s_nomex=[]; e_nomex=[];
            npix_mex=[];   s_mex=[];   e_mex=[];
            npix_omp=[];   s_omp=[];   e_omp=[];            
            for i=1:n_repeats

                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = ...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp] = ...
                    lp.bin_pixels(AB,pix,npix_omp,s_omp,e_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_nomex,npix_omp)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(s_nomex,s_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])

                pix.coordinates = rand(4,n_points);
            end
        end
        function test_mex_nomex_mode4(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex =[];
            npix_omp   = []; s_omp = [];  e_omp =[];

            pix_data = rand(9,n_points);
            pix = PixelDataMemory(pix_data);


            for i= 1:n_repeats
                coord = pix.coordinates;

                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex,pix_ok_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp,pix_ok_omp] = AB.bin_pixels(coord,npix_omp,s_omp,e_omp,pix);


                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_nomex,npix_omp)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(s_nomex,s_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_omp,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])

                pix.data = rand(9,n_points);
            end
        end
        %------------------------------------------------------------------
        function test_mex_nomex_mode3_sigerr_cell(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            %
            [AB,n_points]=obj.prepare_clean_bin_data();

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];

            for i= 1:n_repeats
                coord = rand(4,n_points);
                sig = rand(1,n_points);
                err = rand(1,n_points);

                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,{sig,err});

                config_store.instance.set_value('hor_config','use_mex',true);
                [npix_mex,s_mex,e_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,{sig,err});

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-9,1.e-9])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-9,1.e-9])
            end
        end
        %------------------------------------------------------------------
        function test_proj_mex_nomex_mode2_npix_and_sigerr(obj)

            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end

            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);

            n_repeats = 2;
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();

            npix_nomex = []; s_nomex = []; e_nomex=[];
            npix_mex   = []; s_mex   = []; e_mex=[];
            npix_omp   = []; s_omp   = []; e_omp=[];

            for i=1:n_repeats
                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex] = lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex);


                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex] = lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex);


                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp] = lp.bin_pixels(AB,pix,npix_omp,s_omp,e_omp);


                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_nomex,npix_omp)
                assertEqualToTol(s_nomex,s_mex,'tol',[1e-9 1e-9])
                assertEqualToTol(s_nomex,s_omp,'tol',[1e-9 1e-9])
                assertEqualToTol(e_nomex,e_mex,'tol',[1e-9 1e-9])
                assertEqualToTol(e_nomex,e_omp,'tol',[1e-9 1e-9])

                pix.coordinates = rand(4,n_points);
            end
        end
        function test_mex_nomex_mode2_npix_and_sigerr(obj)

            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [AB,n_points]=obj.prepare_clean_bin_data();

            n_repeats = 2;
            npix_nomex = []; s_nomex = [];e_nomex=[];
            npix_mex   = []; s_mex = [];  e_mex=[];
            npix_omp   = []; s_omp   = []; e_omp=[];

            for i= 1:n_repeats
                pix_data = rand(9,n_points);
                pix = PixelDataMemory(pix_data);
                coord = pix.coordinates;

                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex] = AB.bin_pixels(coord,npix_nomex,s_nomex,e_nomex,pix);


                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                [npix_mex,s_mex,e_mex] = AB.bin_pixels(coord,npix_mex,s_mex,e_mex,pix);


                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                [npix_omp,s_omp,e_omp] = AB.bin_pixels(coord,npix_omp,s_omp,e_omp,pix);

                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_nomex,npix_omp)
                assertEqualToTol(s_nomex,s_mex,'tol',[1e-9 1e-9])
                assertEqualToTol(s_nomex,s_omp,'tol',[1e-9 1e-9])
                assertEqualToTol(e_nomex,e_mex,'tol',[1e-9 1e-9])
                assertEqualToTol(e_nomex,e_omp,'tol',[1e-9 1e-9])
            end
        end
        %------------------------------------------------------------------
        function test_proj_mex_nomex_mode0(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %

            n_repeats = 2;

            [lp,AB,pix,n_points]=obj.prepare_lp_bin_data();

            npix_nomex = [];
            npix_mex   = [];
            npix_omp   = [];

            for i= 1:n_repeats

                config_store.instance.set_value('hor_config','use_mex',false);
                npix_nomex  = lp.bin_pixels(AB,pix,npix_nomex,[],[]);

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);
                npix_mex = lp.bin_pixels(AB,pix,npix_mex,[],[]);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                npix_omp = lp.bin_pixels(AB,pix,npix_omp,[],[]);

                pix.coordinates = rand(4,n_points);
                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_nomex,npix_omp)
            end
        end

        function test_mex_nomex_mode0(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);
            %
            [AB,n_points]=obj.prepare_clean_bin_data();

            n_repeats = 2;
            npix_nomex = [];
            npix_mex   = [];
            npix_omp   = [];

            for i= 1:n_repeats

                in_coord = rand(4,n_points);
                config_store.instance.set_value('hor_config','use_mex',false);

                npix_nomex = AB.bin_pixels(in_coord,npix_nomex);

                config_store.instance.set_value('hor_config','use_mex',true);
                npix_mex = AB.bin_pixels(in_coord,npix_mex);

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                npix_omp = AB.bin_pixels(in_coord,npix_omp);

                assertEqual(npix_nomex,npix_mex)
                assertEqual(npix_nomex,npix_omp)
            end
        end
    end
    methods(Static)
        function [AB,n_points]=prepare_clean_bin_data(nbins_all_dims)
            if nargin == 0
                nbins_all_dims = [50,1,50,1];
            end
            AB = AxesBlockBase_tester('nbins_all_dims',nbins_all_dims, ...
                'img_range',[0,0,0,0;1,0.8,1,0.8]);
            n_points = 2000;
        end

        function [lp,ab,pix,n_points]=prepare_lp_bin_data()
            ab = line_axes('nbins_all_dims',[50,1,50,1], ...
                'img_range',[0,0,0,0;1,0.8,1,0.8]);
            n_points = 5000;
            pix_id = 10;
            pix_coord = rand(9,n_points);
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;
            pix_coord(PixelDataBase.field_index('run_idx'),10:20) = 2*pix_id;

            pix = PixelDataMemory(pix_coord);
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[0,0,0]);
        end
    end
end
