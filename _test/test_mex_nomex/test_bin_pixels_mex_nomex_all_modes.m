classdef test_bin_pixels_mex_nomex_all_modes < TestCase
    % Series of tests to check work of mex files against Matlab files

    properties
        no_mex;
        n_threads = 8;
        n_repeats = 2;
    end

    methods
        function obj=test_bin_pixels_mex_nomex_all_modes(varargin)
            if nargin>0
                name=varargin{1};
            else
                name = 'test_bin_pixels_mex_nomex_all_modes';
            end
            obj = obj@TestCase(name);

            [~,n_errors] = check_horace_mex();
            obj.no_mex = n_errors > 0;
        end
        %==================================================================
        %==================================================================
        function [t_nomex,t_mex,t_omp] = test_proj_mex_nomex_mode8_nosort_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % [npix,s,e,is_sel] =...
            %     lp.bin_pixels(AB,pix,npix,s,e,'-selected_only');

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj performance mode8 (bin pixels + unique runid + return idx + selected):",...
                true, ... debug mode
                lp,AB,pix, ...
                3, ... n_accum
                1, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                true,... sort_pixels
                '-selected_only');
        end

        function [t_nomex,t_mex,t_omp] = test_proj_mex_nomex_mode8_nosort_id_idx_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %[npix,s,e,pix_ok,uniqId,pix_idx,sel] =...
            %         lp.bin_pixels(AB,pix,npix,s,e,uniqId);

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj mex/nomex performance mode8 (bin pixels nosort + unique runid + idx + selected):",...
                true, ... test mode
                lp,AB,pix, ...
                4, ... n_accum
                3, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                true... sort pixels
                );
        end

        function [t_nomex,t_mex,t_omp] = test_mex_nomex_mode8_nosort_idx_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %
            %[npix,s,e,is_sel_nom] = ...
            %        AB.bin_pixels(coord,npix,s,e,pix,'-selected_only');
            [AB,n_points]=obj.prepare_clean_bin_data([50,1,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode8 (bin pixels + selected only):", ...
                true, ... test mode
                AB,3, ... n_accum
                1, ...  n_add_inputs (pix included)
                1, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                true,... sort pixels
                '-selected');
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = test_proj_mex_nomex_mode7_nosort_id_idx(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %[npix,s,e,pix_ok,uniqId,pix_idx] = ...
            %       lp.bin_pixels(AB,pix,npix,s,e,uniqId);

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));
            al_matr = rotvec_to_rotmat([10,20,15]);
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj mex/nomex performance mode7 (bin nosort + unique runid + img idx):", ...
                true, ... test mode
                lp,AB,pix, ...
                4, ... n_accum
                2, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                true,... sort pixels
                al_matr);
        end
        function [t_nomex,t_mex,t_omp] = test_mex_nomex_mode7_nosort_id_idx(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %    [npix,s,e,pix_ok,uniqId,pix_idx] = ...
            %        AB.bin_pixels(coord,npix,s,e,pix,uniqId);

            rng(10)
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode7 (bin pixels(nosort) + unique runid + img idx):", ...
                true, ... test mode
                AB,4, ... n_accum
                1, ...  n_add_inputs
                2, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                true,... sort pixels
                '-nosort');
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = test_mex_nomex_mode6_nosort_id(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %   [npix,s,e,pix_ok,uniqId] = ...
            %        AB.bin_pixels(coord,npix,s,e,pix,uniqId,'-nosort');
            rng(10);
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode6 (bin pixels(nosort) + unique runid):", ...
                true, ... test mode
                AB,4, ... n_accum
                1, ...  n_add_inputs
                1, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                true,... sort pixels
                '-nosort');
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = test_proj_mex_nomex_mode5_sort_and_uid(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end

            %[npix,s,e,pix_ok,uniqId] =...
            %     lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId);

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            al_matr = rotvec_to_rotmat([10,20,15]);
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj mex/nomex performance mode5 (bin and sort pixels + unique runid):", ...
                true, ... test mode
                lp,AB,pix, ...
                4, ... n_accum
                1, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                false, ... sort pixels
                al_matr);
        end

        function [t_nomex,t_mex,t_omp] = test_mex_nomex_mode5_sort_and_uid(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %    [npix,s,e,pix_ok,uniqId] = ...
            %        AB.bin_pixels(coord,npix,s,e,pix,uniqId,'-nosort');

            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode5 (bin and sort pixels + calc unique ID):", ...
                true, ... test mode
                AB,3, ... n_accum
                2, ...  n_add_inputs
                2, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5 ... n_repeats
                ,false... sort pixels
                );
        end

        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = test_proj_mex_nomex_mode4_and_align(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % [npix,s,e,pix_ok] = ...
            %     lp.bin_pixels(AB,pix,npix,s,e);

            %
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            al_matr = rotvec_to_rotmat([10,20,15]);
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj mex/nomex performance mode4 (bin and sort pixels applying alignment):", ...
                true, ... debug mode
                lp,AB,pix, ...
                3, ... n_accum
                1, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                false,... sort pixels
                al_matr);
        end

        function [t_nomex,t_mex,t_omp] = test_mex_nomex_mode4_and_align(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %[npix,s,e,pix_ok] = AB.bin_pixels(coord,npix,s,e,pix);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            al_matr = rotvec_to_rotmat([10,20,15]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode4 (bin and sort aligned pixels):", ...
                true, ... test mode
                AB,3, ... n_accum
                1, ...  n_add_inputs
                1, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                false,... sort pixels
                al_matr);
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = test_proj_mex_nomex_mode0(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %
            %npix_nomex  = lp.bin_pixels(AB,pix,npix_nomex,[],[]);

            [lp,AB,pix,n_points]=obj.prepare_lp_bin_data();
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Mex/nomex proj performance mode0 (bin pix, calc npix):", ...
                true, ... test mode
                lp,AB,pix,1, ... n_accum
                0, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                false... sort pixels
                );
        end
        function [t_nomex,t_mex,t_omp] = test_mex_nomex_mode0(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished
            %
            [AB,n_points]=obj.prepare_clean_bin_data();
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode0 (bin coord, calc npix):", ...
                true, ...test mode
                AB,1, ... n_accum
                0, ...  n_add_inputs
                0, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                obj.n_repeats, ... n_repeats
                false... sort pixels
                );
        end
    end
    methods(Static)
        function [AB,n_points]=prepare_clean_bin_data(nbins_all_dims)
            % prepare input parameters for binning data using AxesBlock class
            if nargin == 0
                nbins_all_dims = [50,1,50,1];
            end
            AB = AxesBlockBase_tester('nbins_all_dims',nbins_all_dims, ...
                'img_range',[0,0,0,0;1,0.8,1,0.8]);
            n_points = 20000 ;% 2000; %
        end

        function [lp,ab,pix,n_points]=prepare_lp_bin_data()
            % prepare input data for binning pixels using projection class
            %
            ab = line_axes('nbins_all_dims',[50,1,50,1], ...
                'img_range',[-1,-1,-1,0;1,0.8,1,0.8]);  % large pix contribution
            %'img_range',[0,0,0,0;1,0.8,1,0.8]);  % low pixel contribution

            %contribution
            n_points = 20000; % 2000; %
            pix_id = 10;
            pix_coord = rand(9,n_points);
            det_ids = 1024+floor(100000*rand(1,n_points));
            en_ids = floor(500*rand(1,n_points));
            pix_coord(PixelDataBase.field_index('run_idx'),:)     = pix_id;
            pix_coord(PixelDataBase.field_index('run_idx'),10:20) = 2*pix_id;
            pix_coord(PixelDataBase.field_index('detector_idx'),:) = det_ids;
            pix_coord(PixelDataBase.field_index('energy_idx'),:)   = en_ids;


            pix = PixelDataMemory(pix_coord);
            lp = line_proj([1,1,0],[0,0,1],'alatt',[1,2,3],'angdeg',[70,80,110],'offset',[0,0,0]);
        end
    end
end
