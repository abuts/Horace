classdef performance_bin_pixels_mex_nomex
    % Series of tests to check performance of mex files against Matlab
    % files

    properties
        this_folder;
        no_mex;
        perf_tests_list
        result_name
        perf_results;
        n_threads;
    end

    methods
        function obj=performance_bin_pixels_mex_nomex(varargin)

            obj.this_folder = fileparts(which('performance_bin_pixels_mex_nomex.m'));
            obj.n_threads = 8;
            [~,n_errors] = check_horace_mex();
            obj.no_mex = n_errors > 0;
            test_names = {...
                'mode8d (proj pixels + selected)',                            ... 8d (dnd)
                'mode8p (proj pixels + unique runid + return idx + selected)',... 8
                'mode7p (proj pixels + unique runid + return idx)',...            7
                'mode5p (proj and sort pixels + unique runid)',...                5
                'mode4p (proj and sort pixels applying alignment)',...            4a
                'mode4p (proj and sort pixels)',...                               4
                'mode2p (proj pixels + sig_err)',...                              2
                'mode0p (proj coord, calc npix)'...                               0
                'mode8 (bin pixels + unique runid + return idx + selected)',... 8
                'mode7 (bin pixels + unique runid + return idx)',...            7
                'mode6 (bin pixels + unique runid)',...                         6
                'mode5 (bin and sort pixels + unique runid)',...                5
                'mode4 (bin and sort pixels applying alignment)',...            4a
                'mode4 (bin and sort pixels)',...                               4
                'mode3 (binning cellarrays of data over coordinate frame)',...  3
                'mode2 (bin pixels + sig_err)',...                              2
                'mode0 (bin coord, calc npix)'...
                };
            test_func = {...
                @()performance_proj_mex_nomex_mode8_nosort_sel(obj),...    8d
                @()performance_proj_mex_nomex_mode8_nosort_id_idx_sel(obj),... 8
                @()performance_proj_mex_nomex_mode7_nosort_id_idx(obj),...  7
                @()performance_proj_mex_nomex_mode5_sort_and_uid(obj),...   5
                @()performance_proj_mex_nomex_mode4_and_align(obj),...      4a
                @()performance_proj_mex_nomex_mode4(obj),...                4
                @()performance_proj_mex_nomex_mode2_npix_and_sigerr(obj),...2
                @()performance_proj_mex_nomex_mode0(obj)...                 0
                @()performance_mex_nomex_mode8_nosort_idx_sel(obj),... 8
                @()performance_mex_nomex_mode7_nosort_id_idx(obj),...  7
                @()performance_mex_nomex_mode6_nosort_id(obj),...      6
                @()performance_mex_nomex_mode5_sort_and_uid(obj),...   5
                @()performance_mex_nomex_mode4_and_align(obj),...      4a
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
                [t_nomex,t_mex,t_omp] = fh();
                mode_name = mode_name{1};
                perf_res.(mode_name) = [t_nomex(:)';t_mex(:)';t_omp];
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
        function [t_nomex,t_mex,t_omp] = performance_proj_mex_nomex_mode8_nosort_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % [npix,s,e,is_sel] =...
            %     lp.bin_pixels(AB,pix,npix,s,e,'-selected_only');
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj performance mode8 (bin pixels + selected):",...
                false, ... debug mode
                lp,AB,pix, ...
                3, ... n_accum
                1, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                10, ... n_repeats
                true,... sort_pixels
                '-selected_only');
        end

        function [t_nomex,t_mex,t_omp] = performance_proj_mex_nomex_mode8_nosort_id_idx_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %[npix,s,e,pix_ok,uniqId,pix_idx,sel] =...
            %         lp.bin_pixels(AB,pix,npix,s,e,uniqId);

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj mex/nomex performance mode8 (bin pixels nosort + unique runid + idx + selected):",...
                false, ... debug mode
                lp,AB,pix, ...
                4, ... n_accum
                3, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                true... sort pixels
                );
            % REFERENCE DATA: ndw2671
            %*** time of first step,    nomex:  1.9(sec)  mex:  1.1(sec); Acceleration :  1.8
            %*** Average time per step, nomex:  1.9(sec)  mex:  1.1(sec); Acceleration :  1.8
        end

        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode8_nosort_idx_sel(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %
            %[npix,s,e,is_sel_nom] = ...
            %        AB.bin_pixels(coord,npix,s,e,pix,'-selected_only');
            [AB,n_points]=obj.prepare_clean_bin_data([50,1,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode8 (bin pixels + selected only):", ...
                false, ... test mode
                AB,3, ... n_accum
                1, ...  n_add_inputs (pix included)
                1, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                true,... sort pixels
                '-selected');
            % end
            % disp_perf_results(t_nomex,t_mex,t_omp);
            % REFERENCE DATA: ndw2671
            %*** Mex/nomex performance mode7 (bin pixels + unique runid + return idx + selected):
            %*** time of first step,    nomex:  3.1(sec)  mex:  1.2(sec); Acceleration :  2.5
            %*** Average time per step, nomex:  3.1(sec)  mex:  1.3(sec); Acceleration :  2.3
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = performance_proj_mex_nomex_mode7_nosort_id_idx(obj)
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
                false, ... test mode
                lp,AB,pix, ...
                4, ... n_accum
                2, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                true,... sort pixels
                al_matr);
            % REFERENCE DATA: ndw2671
            %*** time of first step,    nomex:  1.9(sec)  mex:  1.1(sec); Acceleration :  1.8
            %*** Average time per step, nomex:  1.9(sec)  mex:  1.1(sec); Acceleration :  1.8
        end
        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode7_nosort_id_idx(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %    [npix,s,e,pix_ok,uniqId,pix_idx] = ...
            %        AB.bin_pixels(coord,npix,s,e,pix,uniqId);

            rng(10)
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode7 (bin pixels(nosort) + unique runid + img idx):", ...
                false, ... test mode
                AB,4, ... n_accum
                1, ...  n_add_inputs
                2, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                true,... sort pixels
                '-nosort');
            % REFERENCE DATA: ndw1737
            % *** Mex/nomex performance mode7 (bin pixels + unique runid + return idx):
            % *** first step:    nomex:  2.8(sec)  mex:  1.5(sec);  omp: 0.71(sec); Acc :  1.9/   4/ 2.1
            % *** Avrg per step: nomex:  2.9(sec)  mex:  1.4(sec);  omp: 0.73(sec); Acc :    2/ 3.9/ 1.9
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode6_nosort_id(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %   [npix,s,e,pix_ok,uniqId] = ...
            %        AB.bin_pixels(coord,npix,s,e,pix,uniqId,'-nosort');
            rng(10);
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode6 (bin pixels(nosort) + unique runid):", ...
                false, ... test mode
                AB,4, ... n_accum
                1, ...  n_add_inputs
                1, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                true,... sort pixels
                '-nosort');
            % REFERENCE DATA: ndw2671
            %*** Mex/nomex performance mode6 (bin pixels + unique runid + return idx):
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = performance_proj_mex_nomex_mode5_sort_and_uid(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end

            %[npix,s,e,pix_ok,uniqId] =...
            %     lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId);

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            al_matr = rotvec_to_rotmat([10,20,15]);
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj mex/nomex performance mode5 (bin and sort pixels + unique runid):", ...
                false, ... debug mode
                lp,AB,pix, ...
                4, ... n_accum
                1, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                false, ... sort pixels
                al_matr);
            % REFEFENCE Data NDW2671
            %*** Proj mex/nomex performance mode5 (bin and sort pixels + unique runid):
            %*** time of first step,    nomex:  2.3(sec)  mex:  1.1(sec); Acceleration :  2.1
            %*** Average time per step, nomex:  2.4(sec)  mex:    1(sec); Acceleration :  2.3
            % REFEFENCE Data NDW1737
            %* Proj mex/nomex performance mode5 (bin and sort pixels + unique runid):
            %..........
            %*** first step:    nomex:  4.7(sec)  mex:  2.7(sec);  omp: 0.88(sec); Acc :  1.8/ 5.3/   3
            %*** Avrg per step: nomex:  5.6(sec)  mex:  2.9(sec);  omp: 0.99(sec); Acc :    2/ 5.7/ 2.9
        end
        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode5_sort_and_uid(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %    [npix,s,e,pix_ok,uniqId] = ...
            %        AB.bin_pixels(coord,npix,s,e,pix,uniqId,'-nosort');

            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode5 (bin and sort pixels + calc unique ID):", ...
                false, ... test mode
                AB,3, ... n_accum
                2, ...  n_add_inputs
                2, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5 ... n_repeats
                ,false... sort pixels
                );
            % REFEFENCE Data NDW2671
            %*** Mex/nomex performance mode5 (bin and sort pixels + unique runid):
            %*** time of first step,    nomex:  5.8(sec)  mex:  2.1(sec); Acceleration :  2.7
            %*** Average time per step, nomex:  5.7(sec)  mex:  2.1(sec); Acceleration :  2.7
            % REFEFENCE Data NDW1737
            %*** Mex/nomex performance mode5 (bin and sort pixels + unique runid):
            %..........
            %*** first step:    nomex:  5.5(sec)  mex:    2(sec);  omp:  1.3(sec); Acc :  2.7/ 4.4/ 1.6
            %*** Avrg per step: nomex:  5.4(sec)  mex:  2.1(sec);  omp:  1.3(sec); Acc :  2.6/ 4.3/ 1.6
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = performance_proj_mex_nomex_mode4_and_align(obj)
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
                false, ... debug mode
                lp,AB,pix, ...
                3, ... n_accum
                1, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                false,... sort pixels
                al_matr);
            % REFERENCE Data ndw2671
            %*** Proj mex/nomex performance mode4 (sort pixels):
            %*** time of first step,    nomex:  2.3(sec)  mex:  1.1(sec); Acceleration :  2.1
            %*** Average time per step, nomex:  2.2(sec)  mex:  1.1(sec); Acceleration :  2.1

        end
        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode4_and_align(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %[npix,s,e,pix_ok] = AB.bin_pixels(coord,npix,s,e,pix);
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            al_matr = rotvec_to_rotmat([10,20,15]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode4 (bin and sort aligned pixels):", ...
                false, ... test mode
                AB,3, ... n_accum
                1, ...  n_add_inputs
                1, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                false,... sort pixels
                al_matr);
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode4 (bin and sort pixels applying alignment):
            %*** time of first step,    nomex:  5.9(sec)  mex:  1.7(sec); Acceleration :  3.4
            %*** Average time per step, nomex:  6.1(sec)  mex:  1.9(sec); Acceleration :  3.2
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = performance_proj_mex_nomex_mode4(obj)

            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %[npix,s,e,pix_ok] = ...
            %        lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex);

            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Proj Mex/nomex performance mode4 (bin and sort pixels):", ...
                false, ... debug mode
                lp,AB,pix, ...
                3, ... n_accum
                1, ... n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5, ... n_repeats
                false... sort pixels
                );

            % REFERENCE Data ndw2671
            %*** Proj mex/nomex performance mode2 (bin pixels + sig_err):
            %*** time of first step,    nomex:  1.9(sec)  mex:  0.8(sec); Acceleration :  2.4
            %*** Average time per step, nomex:  1.9(sec)  mex: 0.84(sec); Acceleration :  2.3
        end
        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode4(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %
            [AB,n_points]=obj.prepare_clean_bin_data([50,20,50,20]);
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode4 (bin and sort pixels):", ...
                false, ... test mode
                AB,3, ... n_accum
                1, ...  n_add_inputs
                1, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                5 ... n_repeats
                ,false... sort pixels
                );
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode4 (bin and sort pixels):
            %*** time of first step,    nomex:  5.4(sec)  mex:  1.7(sec); Acceleration :  3.1
            %*** Average time per step, nomex:  5.4(sec)  mex:  1.8(sec); Acceleration :    3
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode3_sigerr_cell(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %
            [AB,n_points]=obj.prepare_clean_bin_data();
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode3 (binning cellarrays of data over coordinate frame):", ...
                false, ... test mode
                AB,3, ... n_accum
                {1}, ...  n_add_inputs
                0, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                10 ... n_repeats
                ,false... sort pixels
                );

            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode3 (binning cellarrays of data over coordinate frame):
            %*** time of first step,    nomex:  2.2(sec)  mex:  0.7(sec); Acceleration :  3.2
            %*** Average time per step, nomex:  2.2(sec)  mex: 0.69(sec); Acceleration :  3.3
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = performance_proj_mex_nomex_mode2_npix_and_sigerr(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %  [npix,s,e] = lp.bin_pixels(AB,pix,npix,s,e);
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Mex/nomex proj performance mode2 (bin pix, calc npix):", ...
                false, ... performance mode
                lp,AB,pix,3, ... n_accum
                0, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                10 ... n_repeats
                );
        end
        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode2_npix_and_sigerr(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %
            [AB,n_points]=obj.prepare_clean_bin_data();
            [t_nomex,t_mex,t_omp] = common_ab_tester("*** Mex/nomex performance mode2 (bin pixels + sig_err):", ...
                false, ... test mode
                AB,3, ... n_accum
                1, ...  n_add_inputs
                0, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                10 ... n_repeats
                ,false... sort pixels
                );
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode2 (bin pixelsbin pixels + sig_err):
            %*** time of first step,    nomex:  3.4(sec)  mex: 0.68(sec); Acceleration :    5
            %*** Average time per step, nomex:  3.4(sec)  mex: 0.71(sec); Acceleration :  4.8
            % REFERENCE Data ndlt1737
            %*** first step:    nomex:  1.1(sec)  mex: 0.24(sec);  omp: 0.041(sec); Acc :  4.4/  26/ 5.8
            %*** Avrg per step: nomex:  1.1(sec)  mex: 0.24(sec);  omp: 0.043(sec); Acc :  4.5/  25/ 5.5
        end
        %------------------------------------------------------------------
        function [t_nomex,t_mex,t_omp] = performance_proj_mex_nomex_mode0(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            %
            %npix_nomex  = lp.bin_pixels(AB,pix,npix_nomex,[],[]);

            [lp,AB,pix,n_points]=obj.prepare_lp_bin_data();
            [t_nomex,t_mex,t_omp] = common_lp_tester( ...
                "*** Mex/nomex proj performance mode0 (bin pix, calc npix):", ...
                false, ... test mode
                lp,AB,pix,1, ... n_accum
                0, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                10 ... n_repeats
                );
            % REFERENCE Data ndlt1737
            %*** first step:    nomex:  2.9(sec)  mex:  2.3(sec);  omp: 0.46(sec); Acc :  1.3/ 6.4
            %*** Avrg per step, nomex:  2.4(sec)  mex: 0.65(sec);  omp: 0.47(sec); Acc :  3.7/ 5.1
        end
        function [t_nomex,t_mex,t_omp] = performance_mex_nomex_mode0(obj)
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished
            %
            [AB,n_points]=obj.prepare_clean_bin_data();
            [t_nomex,t_mex,t_omp] = common_ab_tester( ...
                "*** Mex/nomex performance mode0 (bin coord, calc npix):", ...
                false, ... test mode
                AB,1, ... n_accum
                0, ...  n_add_inputs
                0, ...  n_add_outputs
                obj.n_threads, ...
                n_points, ...
                10 ... n_repeats
                ,false... sort pixels
                );
            % REFERENCE Data ndw2671
            %*** Mex/nomex performance mode0 (bin coord, calc npix):
            %*** time of first step,    nomex:  1.4(sec)  mex: 0.61(sec); Acceleration :  2.3
            %*** Average time per step, nomex:  1.4(sec)  mex: 0.62(sec); Acceleration :  2.2
            % REFERENCE Data ndlt1737
            %*** first step:    nomex: 0.51(sec)  mex: 0.22(sec);  omp: 0.037(sec); Acc :  2.3/  14/ 6.1
            %*** Avrg per step: nomex: 0.51(sec)  mex: 0.22(sec);  omp: 0.037(sec); Acc :  2.3/  14/ 5.9
        end
        function [t_omp1,t_omp] = check_omp_scaling(obj)
            % Method to test how C++ code performance scales with
            % number of OMP threads.
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1,'min_npix_for_omp_cut',0);

            [lp,AB,pix,n_points]=obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));
            al_matr = rotvec_to_rotmat([10,20,15]);

            n_threads_max = 30;
            n_repeats = 5;
            t_omp1 = zeros(1,n_threads_max);
            t_omp  = zeros(1,n_threads_max);
            for ns = 1:n_threads_max
                % Test mosf often used pixel binning method (cut)
                t_omp_all = threads_impact_tester( ...
                    "*** Proj mex/nomex performance mode7 (bin nosort + unique runid + img idx):", ...
                    true, ... test mode
                    lp,AB,pix, ...
                    4, ... n_accum
                    2, ... n_add_outputs
                    ns, ... number of thread to test.
                    n_repeats, ... n_repeats
                    al_matr);
                % Test most ofhen used direct binning method
                % t_omp_all = threads_impact_tester( ...
                %     "*** Mex/nomex proj performance mode2 (bin pix, calc npix):", ...
                %     true, ... test_mode
                %     lp,AB,pix, ...
                %     3, ... n_accum
                %     0, ...  n_add_outputs
                %     ns, ...
                %     n_repeats ... n_repeats
                %     );
                %
                t_omp1(ns) = t_omp_all(1);
                t_omp(ns) = sum(t_omp_all)/n_repeats;
                dt = t_omp_all-t_omp(ns);
                mm = min_max(dt);
                fprintf( ...
                    '*** n_threads = %d; first step:  %4.2g(sec); avrg: %4.2g(sec)%4.2g+%4.2g; Acc : %4.2g/%4.2g\n', ...
                    ns,t_omp1(ns),t_omp(ns),mm(1),mm(2), ...
                    t_omp1(1)/t_omp1(ns),t_omp1(1)/t_omp(ns));
            end

        end
        function [t_nomex,t_mex,t_omp] = profile_proj_mex_nomex_mode8_nosort_id_idx_sel(obj)
            % Detailed performance test for a cpecific performance test
            % Convenient to use while debugging/profiling the code.
            if obj.no_mex
                skipTest('Can not test mex code to check binning against mex');
            end
            % this will recover existing configuration after test have been
            % finished and temporary mex/nomex values will be set within
            % the loop.
            clObHor = set_temporary_config_options(hor_config, 'use_mex', false,'log_level',-1);
            clObPar = set_temporary_config_options(parallel_config, 'threads', 1);
            %
            rng(10);
            [lp,AB,pix,n_points] = obj.prepare_lp_bin_data();
            pix.run_idx  =  500+floor(100*rand(1,n_points));

            n_repeats = 5;
            npix_nomex = []; s_nomex = [];e_nomex=[];uniqId_nom = [];
            npix_mex   = []; s_mex = [];  e_mex=[];uniqId_mex = [];
            npix_omp   = []; s_omp = [];  e_omp=[];uniqId_omp = [];

            t_nomex = zeros(1,n_repeats);
            t_mex  = zeros(1,n_repeats);
            t_omp  = zeros(1,n_repeats);
            disp("*** Proj mex/nomex performance mode8 (bin pixels nosort + unique runid + id+idx+selected):")
            for i= 1:n_repeats
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',false);
                t1 = tic();
                [npix_nomex,s_nomex,e_nomex,pix_ok_nom,uniqId_nom,pix_idx_nom,selected_nom] =...
                    lp.bin_pixels(AB,pix,npix_nomex,s_nomex,e_nomex,uniqId_nom);
                t_nomex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('hor_config','use_mex',true);
                config_store.instance.set_value('parallel_config','threads',1);

                t1 = tic();
                [npix_mex,s_mex,e_mex,pix_ok_mex,uniqId_mex,pix_idx_mex,selected_mex] = ...
                    lp.bin_pixels(AB,pix,npix_mex,s_mex,e_mex,uniqId_mex);
                t_mex(i) = toc(t1);
                fprintf('.')

                config_store.instance.set_value('parallel_config','threads',obj.n_threads);
                t1 = tic();
                [npix_omp,s_omp,e_omp,pix_ok_omp,uniqId_omp,pix_idx_omp,selected_omp] = ...
                    lp.bin_pixels(AB,pix,npix_omp,s_omp,e_omp,uniqId_omp);
                t_omp(i) = toc(t1);
                fprintf('.')

                assertEqual(selected_nom,selected_mex);
                assertEqual(selected_nom,selected_omp);
                assertEqual(sort(int64(pix_idx_nom)),sort(pix_idx_omp));
            end
            disp_perf_results(t_nomex,t_mex,t_omp);

            assertEqual(uint32(uniqId_nom),uniqId_mex);
            assertEqual(uint32(uniqId_nom),uniqId_omp);
            assertEqual(int64(pix_idx_nom),pix_idx_mex);
            assertEqual(selected_nom,selected_mex);
            assertEqual(selected_nom,selected_omp);

            assertEqual(npix_nomex,npix_mex)
            assertEqualToTol(s_nomex,s_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(e_nomex,e_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(pix_ok_nom,pix_ok_mex,'tol',[1.e-12,1.e-12])
            assertEqualToTol(pix_ok_nom,pix_ok_omp,'tol',[1.e-12,1.e-12])
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
            n_points = 20000000;% 2000; %
        end

        function [lp,ab,pix,n_points]=prepare_lp_bin_data()
            % prepare input data for binning pixels using projection class
            %
            ab = line_axes('nbins_all_dims',[50,1,50,1], ...
                'img_range',[-1,-1,-1,0;1,0.8,1,0.8]);  % large pix contribution
            %'img_range',[0,0,0,0;1,0.8,1,0.8]);  % low pixel contribution

            %contribution
            n_points = 20000000; % 2000; %
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
