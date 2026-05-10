classdef test_bin_pixels_line_proj_mex_nomex < TestCase
    % Series of tests to check work of mex files against Matlab files

    properties
        this_folder;
        no_mex;

        nPolar=99;
        nAzim =101;
        nDet;
        nEn  = 102;
        efix=100;
    end

    methods
        function obj=test_bin_pixels_line_proj_mex_nomex(varargin)
            if nargin>0
                name=varargin{1};
            else
                name = 'test_bin_pixels_line_proj_mex_nomex';
            end
            obj = obj@TestCase(name);

            obj.this_folder = fileparts(which('test_bin_pixels_line_proj_mex_nomex.m'));
            obj.nDet=obj.nPolar*obj.nAzim;

            [~,n_errors] = check_horace_mex();
            obj.no_mex = n_errors > 0;
        end

        function obj=test_bin_pixels_mex_multithread(obj)
            if obj.no_mex
                skipTest('Can not use and test mex code to bin pixels in parallel');
            end

            [data,pix]=gen_fake_accum_cut_data(obj,[1,0,0],[0,1,0]);

            clObHor = set_temporary_config_options(hor_config, 'use_mex', true);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1);

            [npix_1,s_1,e_1,pix_ok_1,unique_runid_1] = ...
                data.proj.bin_pixels(data.axes,pix,[],[],[]);

            clear clObPar;
            clObPar = set_temporary_config_options(parallel_config, 'threads', 8);
            [npix_8,s_8,e_8,pix_ok_8,unique_runid_8] = ...
                data.proj.bin_pixels(data.axes,pix,[],[],[]);

            assertEqual(npix_1,npix_8)
            assertEqual(s_1,s_8)
            assertEqual(e_1,e_8)
            assertEqual(pix_ok_1,pix_ok_8)
            assertEqual(unique_runid_1,unique_runid_8)
        end

        function test_bin_pixels_with_unique_id_mex_nomex_two_pages(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            AB = line_axes('nbins_all_dims',[50,1,50,1], ...
                'img_range',[0,0,0,0;1,0.8,1,0.8]);
            n_points = 200;

            pix_coord = rand(9,n_points);
            pix_id = 500+floor(100*rand(1,n_points));
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;

            pix = PixelDataMemory(pix_coord);
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[0,0,0]);

            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            for i=1:2
                config_store.instance.set_value('hor_config','use_mex',false);
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] =...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);

                config_store.instance.set_value('hor_config','use_mex',true);
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);

                assertEqual(uint32(uniqId_nom),uniqId_mex);

                assertEqual(npix_nomex,npix_mex)
                assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
                assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
                pix.coordinates = rand(4,n_points);
            end
        end

        function test_bin_pixels_single_with_unique_id_mex_nomex(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            AB = line_axes('nbins_all_dims',[50,1,50,1], ...
                'img_range',[0,0,0,0;1,0.8,1,0.8]);
            n_points = 200;

            pix_coord = single(rand(9,n_points));
            pix_id = single(500+floor(100*rand(1,n_points)));
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;

            pix = PixelDataMemory(pix_coord);
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[0,0,0]);

            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            config_store.instance.set_value('hor_config','use_mex',false);
            [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] =...
                lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);

            config_store.instance.set_value('hor_config','use_mex',true);
            [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] = ...
                lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);

            assertEqual(uint32(uniqId_nom),uniqId_mex);

            assertEqual(npix_nomex,npix_mex)
            assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
        end

        function test_bin_pixels_with_offset_mex_nomex(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            AB = line_axes('nbins_all_dims',[50,1,50,1], ...
                'img_range', [...
                -1,-0.5,-1, .0; ...
                0  , 1, 1,0.8]);
            n_points = 200;

            pix_coord = rand(9,n_points);
            pix = PixelDataMemory(pix_coord);
            pix.run_idx = 500+floor(100*rand(1,n_points));
            lp = line_proj([1,1,0],[0,0,1],'alatt',[2,2,2],'angdeg',[70,80,110],'offset',[1,1,0]);

            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            config_store.instance.set_value('hor_config','use_mex',false);
            [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] =...
                lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);

            config_store.instance.set_value('hor_config','use_mex',true);
            [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] = ...
                lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);

            assertEqual(uint32(uniqId_nom),uniqId_mex);

            assertEqual(npix_nomex,npix_mex)
            assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
        end



        function test_bin_pixels_with_unique_id_mex_nomex(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            %
            AB = line_axes('nbins_all_dims',[50,1,50,1], ...
                'img_range',[0,0,0,0;1,0.8,1,0.8]);
            n_points = 200;

            pix_coord = rand(9,n_points);
            pix_id = 500+floor(100*rand(1,n_points));
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;

            pix = PixelDataMemory(pix_coord);
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[0,0,0]);

            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];

            config_store.instance.set_value('hor_config','use_mex',false);
            [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom] =...
                lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);

            config_store.instance.set_value('hor_config','use_mex',true);
            [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex] = ...
                lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);

            assertEqual(uint32(uniqId_nom),uniqId_mex);

            assertEqual(npix_nomex,npix_mex)
            assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
        end

        function obj=test_bin_pixels_on_line_proj_mex_nomex(obj)
            if obj.no_mex
                skipTest('Can not use and test mex code to bin pixels on line proj');
            end

            [data,pix]=gen_fake_accum_cut_data(obj,[1,0,0],[0,1,0]);
            %[v,sizes,rot_ustep,trans_bott_left,ebin,trans_elo,urange_step_pix,urange_step]=gen_fake_accum_cut_data(this,0,0);

            hc = hor_config;
            hc.saveable = false;

            %check matlab-part
            hc.use_mex = false;
            [npix_m,s_m,e_m,pix_ok_m,unique_runid_m] = ...
                data.proj.bin_pixels(data.axes,pix,[],[],[]);

            %check C-part
            hc.use_mex = true;
            [npix_c,s_c,e_c,pix_ok_c,unique_runid_c] = ...
                data.proj.bin_pixels(data.axes,pix,[],[],[]);

            % verify results against each other.
            assertElementsAlmostEqual(npix_m,npix_c,'absolute',1.e-12);
            assertElementsAlmostEqual(s_m,s_c);
            assertElementsAlmostEqual(e_m,e_c);
            assertElementsAlmostEqual(npix_m,npix_c,'absolute',1.e-12);
            assertEqualToTol(pix_ok_m,pix_ok_c);
            assertElementsAlmostEqual(unique_runid_m,double(unique_runid_c));
        end

        function test_return_inputs_mex_mode6_nosort_2D(obj)
            % bin pixels and sort pixels, input/output parameters
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            clObHor = set_temporary_config_options(hor_config, 'use_mex', true);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 8,'min_npix_for_omp_cut',0);

            AB = AxesBlockBase_tester('nbins_all_dims',[10,1,1,40], ...
                'img_range',[-1,-2,-3,-10;1,2,3,40]);
            pix_id = [10,10,11,11,7, 5,5,5,10,10];
            pix_coord = rand(9,10);
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;
            pix = PixelDataMemory(pix_coord);

            in_coord = [];
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[1,1,0]);
            [npix,s,e,pix_ok,unique_id,pix_idx,out_data] = lp.bin_pixels(AB,pix,[],[],[],'-test_mex_inputs');

            assertEqual(size(npix),[10,40]);
            assertEqual(npix,zeros(10,40));
            assertEqual(s,npix);
            assertEqual(e,npix);
            assertTrue(isempty(unique_id));
            assertTrue(isa(unique_id,'uint32'));
            assertTrue(isempty(pix_idx));
            assertTrue(isa(pix_idx,'int64'));


            assertEqual(pix_ok.data,out_data.pix_ok_data);
            assertEqual(pix_ok.data,pix_coord);
            assertEqual(pix_ok.data_range,out_data.pix_ok_data_range);
            % range matrix have been allocated and probably contains zeros
            % but this is not guaranteed.
            assertEqual(size(out_data.pix_ok_data_range),[2,9]);

            assertEqual(out_data.coord_in,in_coord);
            assertEqual(out_data.binning_mode,bin_mode.nosort);
            assertEqual(out_data.num_threads, ...
                config_store.instance().get_value('parallel_config','threads'));
            assertEqual(out_data.data_range,AB.img_range)
            assertEqual(out_data.bins_all_dims,uint32(AB.nbins_all_dims));
            assertTrue(isempty(out_data.unique_runid));
            assertFalse(out_data.force_double);
            assertTrue(out_data.test_input_parsing);
            assertTrue(isempty(out_data.alignment_matr));
            assertEqual(out_data.pix_candidates,pix.data);
            assertTrue(out_data.check_pix_selection);
            assertEqual(out_data.pix_img_idx,pix_idx);

            [mat,off] = lp.get_pix_img_transformation(3);
            assertEqual(out_data.u_offset,off(1:3)')
            assertEqualToTol(out_data.q_to_img,mat,1.e-11);

        end

        function test_return_inputs_mex_mode5_sort_and_unique_id_2D(obj)
            % bin pixels and sort pixels, input/output parameters
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            clObHor = set_temporary_config_options(hor_config, 'use_mex', true);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 8,'min_npix_for_omp_cut',0);

            AB = AxesBlockBase_tester('nbins_all_dims',[10,1,1,40], ...
                'img_range',[-1,-2,-3,-10;1,2,3,40]);
            pix_id = [10,10,11,11,7, 5,5,5,10,10];
            pix_coord = rand(9,10);
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;
            pix = PixelDataMemory(pix_coord);

            in_coord = [];
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[1,1,0]);
            [npix,s,e,pix_ok,unique_id,out_data] = lp.bin_pixels(AB,pix,[],[],[],'-test_mex_inputs');


            assertEqual(size(npix),[10,40]);
            assertEqual(npix,zeros(10,40));
            assertEqual(s,npix);
            assertEqual(e,npix);
            assertTrue(isempty(unique_id));
            assertTrue(isa(unique_id,'uint32'));


            assertEqual(pix_ok.data,out_data.pix_ok_data);
            assertEqual(pix_ok.data,pix_coord);
            assertEqual(pix_ok.data_range,out_data.pix_ok_data_range);
            % range matrix have been allocated and probably contains zeros
            % but this is not guaranteed.
            assertEqual(size(out_data.pix_ok_data_range),[2,9]);

            assertEqual(out_data.coord_in,in_coord);
            assertEqual(out_data.binning_mode,bin_mode.sort_and_uid);
            assertEqual(out_data.num_threads, ...
                config_store.instance().get_value('parallel_config','threads'));
            assertEqual(out_data.data_range,AB.img_range)
            assertEqual(out_data.bins_all_dims,uint32(AB.nbins_all_dims));
            assertTrue(isempty(out_data.unique_runid));
            assertFalse(out_data.force_double);
            assertTrue(out_data.test_input_parsing);
            assertTrue(isempty(out_data.alignment_matr));
            assertEqual(out_data.pix_candidates,pix.data);
            assertTrue(out_data.check_pix_selection);

            [mat,off] = lp.get_pix_img_transformation(3);
            assertEqual(out_data.u_offset,off(1:3)')
            assertEqualToTol(out_data.q_to_img,mat,1.e-11);
        end

        function test_return_inputs_mex_mode4_2D_sort_pix(obj)
            % bin pixels and sort pixels, input/output parameters
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            clObHor = set_temporary_config_options(hor_config, 'use_mex', true);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 8,'min_npix_for_omp_cut',0);

            AB = AxesBlockBase_tester('nbins_all_dims',[10,1,1,40], ...
                'img_range',[-1,-2,-3,-10;1,2,3,40]);
            pix_coord = rand(9,10);
            pix = PixelDataMemory(pix_coord);

            in_coord = [];
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[1,1,0]);
            [npix,s,e,pix_ok,out_data] = lp.bin_pixels(AB,pix,[],[],[],'-test_mex_inputs');


            assertEqual(size(npix),[10,40]);
            assertEqual(npix,zeros(10,40));
            assertEqual(s,npix);
            assertEqual(e,npix);

            assertEqual(pix_ok.data,out_data.pix_ok_data);
            assertEqual(pix_ok.data,pix_coord);
            assertEqual(pix_ok.data_range,out_data.pix_ok_data_range);
            % range matrix have been allocated and probably contains zeros
            % but this is not guaranteed.
            assertEqual(size(out_data.pix_ok_data_range),[2,9]);

            assertEqual(out_data.coord_in,in_coord);
            assertEqual(out_data.binning_mode,bin_mode.sort_pix);
            assertEqual(out_data.num_threads, ...
                config_store.instance().get_value('parallel_config','threads'));
            assertEqual(out_data.data_range,AB.img_range)
            assertEqual(out_data.bins_all_dims,uint32(AB.nbins_all_dims));
            assertTrue(isempty(out_data.unique_runid));
            assertFalse(out_data.force_double);
            assertTrue(out_data.test_input_parsing);
            assertTrue(isempty(out_data.alignment_matr));
            assertEqual(out_data.pix_candidates,pix.data);
            assertTrue(out_data.check_pix_selection);

            [mat,off] = lp.get_pix_img_transformation(3);
            assertEqual(out_data.u_offset,off(1:3)')
            assertEqualToTol(out_data.q_to_img,mat,1.e-11);
        end

        function test_return_inputs_mex_mode2_sig_err_2D(obj)
            % bin pixels no pixel sorting, input/output parameters
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            clObHor = set_temporary_config_options(hor_config, 'use_mex', true);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 8,'min_npix_for_omp_cut',0);

            AB = AxesBlockBase_tester('nbins_all_dims',[10,20,30,40], ...
                'img_range',[-1,-2,-3,-10;1,2,3,40]);
            pix_coord = rand(9,10);
            pix = PixelDataMemory(pix_coord);

            in_coord = [];
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[1,1,0]);
            [npix,s,e,out_data] = lp.bin_pixels(AB,pix,[],[],[],'-test_mex_inputs');

            assertEqual(size(npix),[10,20,30,40]);
            assertEqual(npix,zeros(10,20,30,40));
            assertEqual(s,npix);
            assertEqual(e,npix);

            assertEqual(out_data.coord_in,in_coord);
            assertEqual(out_data.binning_mode,bin_mode.sig_err);
            assertEqual(out_data.num_threads, ...
                config_store.instance().get_value('parallel_config','threads'));
            assertEqual(out_data.data_range,AB.img_range)
            assertEqual(out_data.bins_all_dims,uint32(AB.nbins_all_dims));
            assertTrue(isempty(out_data.unique_runid));
            assertFalse(out_data.force_double);
            assertTrue(out_data.test_input_parsing);
            assertTrue(isempty(out_data.alignment_matr));
            assertEqual(out_data.pix_candidates,pix.data);
            assertTrue(out_data.check_pix_selection);

            [mat,off] = lp.get_pix_img_transformation(3);
            assertEqual(out_data.u_offset,off(1:3)')
            assertEqualToTol(out_data.q_to_img,mat,1.e-11);

        end

        function test_return_inputs_line_proj_mex_mode0(obj)
            % bin pixels and sort pixels, input/output parameters
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            clObHor = set_temporary_config_options(hor_config, 'use_mex', true);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 8,'min_npix_for_omp_cut',0);

            AB = AxesBlockBase_tester('nbins_all_dims',[10,1,1,40], ...
                'img_range',[-1,-2,-3,-10;1,2,3,40]);
            pix_id = [10,10,11,11,7, 5,5,5,10,10];
            pix_coord = rand(9,10);
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;
            pix = PixelDataMemory(pix_coord);

            in_coord = [];
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[1,1,0]);
            [npix,out_data] = lp.bin_pixels(AB,pix,[],[],[],'-test_mex_inputs');

            assertEqual(size(npix),[10,40]);
            assertEqual(npix,zeros(10,40));
            %
            assertEqual(out_data.coord_in,in_coord);
            assertEqual(out_data.binning_mode,bin_mode.npix_only);
            assertEqual(out_data.num_threads, ...
                config_store.instance().get_value('parallel_config','threads'));
            assertEqual(out_data.data_range,AB.img_range)
            assertEqual(out_data.bins_all_dims,uint32(AB.nbins_all_dims));
            assertTrue(isempty(out_data.unique_runid));
            assertFalse(out_data.force_double);
            assertTrue(out_data.test_input_parsing);
            assertTrue(isempty(out_data.alignment_matr));
            assertEqual(out_data.pix_candidates,pix.data);
            assertTrue(out_data.check_pix_selection);


            [mat,off] = lp.get_pix_img_transformation(3);
            assertEqual(out_data.u_offset,off(1:3)')
            assertEqualToTol(out_data.q_to_img,mat,1.e-11);
        end

        function test_return_inputs_line_proj_mex_mode7_nosort_and_selected(obj)
            % bin pixels and sort pixels, input/output parameters
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            clObHor = set_temporary_config_options(hor_config, 'use_mex', true);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 8,'min_npix_for_omp_cut',0);

            AB = AxesBlockBase_tester('nbins_all_dims',[10,1,1,40], ...
                'img_range',[-1,-2,-3,-10;1,2,3,40]);
            pix_id = [10,10,11,11,7, 5,5,5,10,10];
            pix_coord = rand(9,10);
            pix_coord(PixelDataBase.field_index('run_idx'),:) = pix_id;
            pix = PixelDataMemory(pix_coord);

            in_coord = [];
            lp = line_proj('alatt',2.38,'angdeg',90);
            [npix,s,e,pix_ok,unique_id,pix_idx,is_selected,out_data] = lp.bin_pixels(AB,pix,[],[],[],'-test_mex_inputs');

            assertEqual(size(npix),[10,40]);
            assertEqual(npix,zeros(10,40));
            assertEqual(s,npix);
            assertEqual(e,npix);
            assertTrue(isempty(unique_id));
            assertTrue(isa(unique_id,'uint32'));
            assertTrue(isempty(pix_idx));
            assertTrue(isa(pix_idx,'int64'));
            assertTrue(isempty(is_selected));
            assertTrue(isa(is_selected,'logical'));

            assertEqual(pix_ok.data,out_data.pix_ok_data);
            assertEqual(pix_ok.data,pix_coord);
            assertEqual(pix_ok.data_range,out_data.pix_ok_data_range);
            % range matrix have been allocated and probably contains zeros
            % but this is not guaranteed.
            assertEqual(size(out_data.pix_ok_data_range),[2,9]);

            assertEqual(out_data.coord_in,in_coord);
            assertEqual(out_data.binning_mode,bin_mode.nosort_sel);
            assertEqual(out_data.num_threads, ...
                config_store.instance().get_value('parallel_config','threads'));
            assertEqual(out_data.data_range,AB.img_range)
            assertEqual(out_data.bins_all_dims,uint32(AB.nbins_all_dims));
            assertTrue(isempty(out_data.unique_runid));
            assertFalse(out_data.force_double);
            assertTrue(out_data.test_input_parsing);
            assertTrue(isempty(out_data.alignment_matr));
            assertEqual(out_data.pix_candidates,pix.data);
            assertTrue(out_data.check_pix_selection);
            assertEqual(out_data.pix_img_idx,pix_idx);
            assertEqual(out_data.is_pix_selected,is_selected);
            assertEqual(out_data.u_offset,[0,0,0])
            bm = lp.bmatrix();
            trel = 1/bm(1);
            assertEqualToTol(out_data.q_to_img,[trel ,trel ,trel ],1.e-14);
        end

    end
    methods(Access=protected)
        function  rd = calc_fake_data(obj)
            rd = rundatah();
            rd.efix = obj.efix;
            rd.emode=1;
            lat = oriented_lattice(struct('alatt',[1,1,1],'angdeg',[92,88,73],...
                'u',[1,0,0],'v',[1,1,0],'psi',20));
            rd.lattice = lat;

            det = struct('filename','','filepath','');
            det.x2  = ones(1,obj.nDet);
            det.group = 1:obj.nDet;
            polar=(0:(obj.nPolar-1))*(pi/(obj.nPolar-1));
            azim=(0:(obj.nAzim-1))*(2*pi/(obj.nAzim-1));
            det.phi = reshape(repmat(azim,obj.nPolar,1),1,obj.nDet);
            det.azim =reshape(repmat(polar,obj.nAzim,1)',1,obj.nDet);
            det.width= 0.1*ones(1,obj.nAzim*obj.nPolar);
            det.height= 0.1*ones(1,obj.nAzim*obj.nPolar);
            rd.det_par = det;

            S  = rand(obj.nEn,obj.nDet);
            rd.S = S;
            rd.ERR = sqrt(S);
            rd.en =(-obj.efix+(0:(obj.nEn))*(1.99999*obj.efix/(obj.nEn)))';
        end

        function [data,pix]=gen_fake_accum_cut_data(obj,u,v)
            % build fake data to test accumulate cut

            nPixels = obj.nDet*obj.nEn;
            ebin=1.99*obj.efix/obj.nEn;
            en = -obj.efix+(0:(obj.nEn-1))*ebin;

            L1=20;
            L2=10;
            L3=2;
            E0=min(en);
            E1=max(en);
            Es=2;
            proj = line_proj(u,v,'alatt',[3,4,5],'angdeg',[90,90,90]);
            ab = line_axes([0,1,L1],[0,1,L2],[0,0.1,L3],[E0,Es,E1]);
            data = DnDBase.dnd(ab,proj);

            vv=ones(9,nPixels);
            for i=1:3
                p=data.p{i};
                ac=0.5*(p(2:end)+p(1:end-1));
                p_mi=min(ac);
                p_ma=max(ac);
                step=(p_ma-p_mi)/(nPixels-1);
                vv(i,:) =p_mi:step:p_ma;
            end
            vv(4,:)=repmat(en,1,obj.nDet);

            pix = PixelDataBase.create(vv);
        end
    end
end
