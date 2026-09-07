classdef IX_exper_common_test <  TestCase
    %Test class to test IX_experiment constructor and methods
    %

    properties(Constant)
        par_val = {'my_file',6666,10,[1,2,4,8]',70,5,5,5,5};
    end

    methods
        function obj=IX_exper_common_test(name)
            obj = obj@TestCase(name);
        end
    end
    methods(Static,Access=protected)
        function [data,run_id] = build_IX_array_blocks(n_elements,n_blocks,legacy_run_id)
            if nargin<3
                legacy_run_id = true;
            end
            data = cell(n_blocks,1);
            ids  = cell(n_blocks,1);
            unr = [];
            for i=1:n_blocks
                [data{i},ids{i}]=test_IX_experiment.build_IX_array(n_elements,legacy_run_id);
                if legacy_run_id
                    unr1 = unique([ids{i},unr]);
                    while numel(unr1) ~= i*n_elements
                        [data{i},ids{i}]=test_IX_experiment.build_IX_array(n_elements,legacy_run_id);
                        unr1 = unique([ids{i},unr]);
                    end
                    unr = unr1;
                end
            end
            run_id = [ids{:}];
        end

        function expd= build_element(expd)
            % build single random IX_experiment dataset
            par_names={...
                'filename', 'run_id', 'efix','en',...
                'psi','omega','dpsi','gl','gs'};
            expd.do_check_combo_arg = false;
            par_val = IX_exper_common_test.par_val;
            for j=1:numel(par_names)
                if ischar(par_val{j})
                    val = build_tmp_file_name('nxspe_file','');
                elseif numel(par_val{j})>1
                    expd.efix = expd.efix+5;
                    val = sort(rand(size(par_val{j}))*expd.efix);
                else
                    val = round(rand()*par_val{j});
                end

                expd.(par_names{j}) = val;
            end
            expd.filepath = 'some_file_path';
            expd.emode = 1;
            expd.do_check_combo_arg = true;
            expd = expd.check_combo_arg();
        end

        function [data,run_id] = build_IX_array(n_elements,legacy_run_id)
            if nargin<2
                legacy_run_id = false;
            end
            data = repmat(IX_experiment,1,n_elements);
            for i=1:n_elements
                expd = data(i);
                data(i) = IX_exper_common_test.build_element(expd);
            end

            if legacy_run_id
                % ensure run_id are unique to avoid random tests failures
                run_id = arrayfun(@(x)x.run_id,data);
                uniq_id = unique(run_id);
                was_nonunique = false;
                while numel(uniq_id) ~= numel(run_id)
                    was_nonunique = true;
                    run_id  = round(rand(1,n_elements)*IX_exper_common_test.par_val{2});
                    uniq_id = unique(run_id);
                end
                if was_nonunique
                    for i=1:n_elements
                        data(i).run_id = run_id(i);
                    end
                end
            else
                run_id = zeros(1,n_elements);
                % ensure some run-id(s) are duplicated
                data(1).run_id = data(2).run_id;
                % this should be check that no identical IX_experiment-s
                % were generated but it may become cumbersome. Bearing in
                % mind that IX_experiment is build from 8(7 exluding
                % run_id) random properties, we decide to ignore such
                % possibility. 
                for i=1:n_elements
                    data(i).ixexper_id = i;
                    run_id(i) = data(i).run_id;
                end
            end
        end
    end
end
