classdef test_IX_exper_process_addruns <  IX_exper_common_test
    %Test class to test IX_experiment constructor and methods
    %

    properties
    end

    methods
        function obj=test_IX_exper_process_addruns(varargin)
            if nargin == 0
                name = 'test_IX_exper_process_addruns';
            else
                name = varargin{1};
            end
            obj = obj@IX_exper_common_test(name);
        end
        %
        %==================================================================
        function test_add_partially_overlaping_runs(~)
            data1 = test_IX_experiment.build_IX_array(10,false);
            data2 = test_IX_experiment.build_IX_array(5,false);
            data = [data1(1:5),data2(1:5)];
            for i=1:10
                data(i).ixexper_id = i;
            end


            pr_hashes = repmat({''},1,2*numel(data));
            present_runs = cell(1,2*numel(data));
            this_runid_map = fast_map();
            this_runid_map.trivial_map = true;
            n_found_runs = 0;
            [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map,n_subst]=...
                process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,data1,true);
            assertFalse(any(skip_runs));
            assertEqual(n_subst,0);
            assertEqual(size(subst_map),[2,10])

            [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map,n_subst]=...
                process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,data,true);

            assertTrue(all(skip_runs(1:5)));
            assertFalse(any(skip_runs(6:10)));
            assertEqual(n_found_runs,15);
            ref_map = fast_map(uint32(1:15),1:15,true);
            assertEqual(ref_map,this_runid_map);
            pr = [present_runs{:}];
            assertEqual(pr,[data1,data2]);
            assertEqual(size(subst_map),[2,10]);
            assertEqual(n_subst,10);
            assertEqual(subst_map,[1:10;[1:5,11:15]]);

            ref_hashes = cell(1,15);
            for i=1:15
                [~,ref_hashes{i}] = build_hash(present_runs{i});
            end
            assertEqual(pr_hashes(1:15),ref_hashes);
        end

        function test_process_ignore_single_legacy_runs_works(~)
            [data,fids] = test_IX_experiment.build_IX_array(10,true);

            ref_hashes = cell(1,numel(data));
            present_runs = cell(1,2*numel(data));
            pr_hashes = repmat({''},1,2*numel(data));
            for i=1:numel(data)
                [data(i),ref_hashes{i}] = data(i).build_hash();
                pr_hashes{i} = ref_hashes{i};
                present_runs{i} = data(i);
            end
            this_runid_map = fast_map(uint32(fids),1:10);

            n_found_runs = 10;
            [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map,n_el]=...
                process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,data,false);

            assertTrue(all(skip_runs));
            assertEqual(n_found_runs,10);
            ref_map = fast_map(uint32(fids),1:10);
            assertEqual(ref_map,this_runid_map);
            pr = [present_runs{:}];
            assertEqual(pr,data);
            assertEqual(n_el,10);
            assertEqual(subst_map,[flds;flds]);

            assertEqual(pr_hashes(1:10),ref_hashes);
        end

        function test_process_add_single_runs_works(~)
            data = test_IX_experiment.build_IX_array(10,false);

            ref_hashes = cell(1,numel(data));
            for i=1:numel(data)
                [data(i),ref_hashes{i}] = data(i).build_hash();
            end

            present_runs = cell(1,numel(data));
            pr_hashes = repmat({''},1,numel(data));
            [present_runs{1},pr_hashes{1}] = build_hash(data(1));
            n_found_runs = 1;
            this_runid_map = fast_map();
            this_runid_map = this_runid_map.add(data(1).ixexper_id,1);
            this_runid_map.trivial_map = true;

            [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map,n_subst]=...
                process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,data(2:end),false);

            assertFalse(any(skip_runs));
            assertEqual(n_found_runs,10);
            ref_map = fast_map(uint32(1:10),1:10,true);
            assertEqual(ref_map,this_runid_map);
            pr = [present_runs{:}];
            assertEqual(pr,data);
            assertEqual(n_subst,0);
            assertEqual(size(subst_map),[2,9]);


            assertEqual(pr_hashes,ref_hashes);
        end

        function test_process_add_single_legacy_runs_works(~)
            [data,fids] = test_IX_experiment.build_IX_array(10,true);

            ref_hashes = cell(1,numel(data));
            for i=1:numel(data)
                [data(i),ref_hashes{i}] = data(i).build_hash();
            end

            present_runs = cell(1,numel(data));
            pr_hashes = repmat({''},1,numel(data));
            n_found_runs = 0;
            this_runid_map = fast_map();

            [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map,n_subst]=...
                process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,data,false);

            assertTrue(all(~skip_runs));
            assertEqual(n_found_runs,10);
            ref_map = fast_map(uint32(fids),1:10);
            assertEqual(ref_map,this_runid_map);
            pr = [present_runs{:}];
            assertEqual(pr,data);
            assertEqual(n_subst,0);
            assertEqual(size(subst_map),[2,10]);

            assertEqual(pr_hashes,ref_hashes);
        end
    end
end
