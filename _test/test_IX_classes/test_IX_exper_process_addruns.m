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
            [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map]=...
                process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,data,false);

            assertTrue(all(skip_runs));            
            assertEqual(n_found_runs,10);
            ref_map = fast_map(uint32(fids),1:10);
            assertEqual(ref_map,this_runid_map);
            pr = [present_runs{:}];
            assertEqual(pr,data);
            assertTrue(isempty(subst_map));

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

            [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map]=...
                process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,data(2:end),false);

            assertTrue(all(~skip_runs));            
            assertEqual(n_found_runs,10);
            ref_map = fast_map(uint32(1:10),1:10);
            assertEqual(ref_map,this_runid_map);
            pr = [present_runs{:}];
            assertEqual(pr,data);
            assertTrue(isempty(subst_map));

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
            [present_runs{1},pr_hashes{1}] = build_hash(data(1));
            n_found_runs = 1;
            this_runid_map = fast_map();
            this_runid_map = this_runid_map.add(data(1).ixexper_id,1);

            [present_runs,pr_hashes,this_runid_map,n_found_runs,skip_runs,subst_map]=...
                process_addruns(present_runs,pr_hashes,n_found_runs,this_runid_map,data(2:end),false);

            assertTrue(all(~skip_runs));
            assertEqual(n_found_runs,10);
            ref_map = fast_map(uint32(fids),1:10);
            assertEqual(ref_map,this_runid_map);
            pr = [present_runs{:}];
            assertEqual(pr,data);
            assertTrue(isempty(subst_map));

            assertEqual(pr_hashes,ref_hashes);
        end
    end
end
