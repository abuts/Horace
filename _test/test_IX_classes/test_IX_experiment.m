classdef test_IX_experiment <  IX_exper_common_test
    %Test class to test IX_experiment constructor and methods
    %

    properties
    end

    methods
        function obj=test_IX_experiment(varargin)
            if nargin == 0
                name = 'test_IX_experiment';
            else
                name = varargin{1};
            end
            obj =obj@IX_exper_common_test(name);
        end
        %
        function test_hashable_prop(~)
            exper = test_IX_experiment.build_IX_array(1,true);
            hashable_obj_tester(exper);
        end
        %==================================================================
        function test_combine_single_runs_eq_headers_skip_duplicate(~)
            data = test_IX_experiment.build_IX_array(10);
            data(2) = data(7);
            Input = num2cell(data);

            [result,skipped_inputs,this_ixexperid_map,renum_map] = IX_experiment.combine(Input,true);

            hash_defined = arrayfun(@(x)(x.hash_defined),result);
            assertTrue(all(hash_defined));
            assertTrue(isempty(renum_map));

            data = [data(1:6),data(8:10)];

            assertEqual(data,result);
            assertTrue(iscell(skipped_inputs))
            skipped_inputs = [skipped_inputs{:}];
            assertEqual(numel(skipped_inputs),10);
            assertTrue(skipped_inputs(7)); % 7th skipped
            assertFalse(all(skipped_inputs(1:5))); % left itact
            assertFalse(all(skipped_inputs(7:9))); % left itact

            keys = this_ixexperid_map.keys();
            for i=1:numel(keys)
                id = this_ixexperid_map.get(keys(i));
                assertEqual(uint32(result(id).ixexper_id),keys(i));
            end
        end

        %------------------------------------------------------------------
        function test_combine_multirun_same_headers_works(~)
            Input = test_IX_experiment.build_IX_array_blocks(10,3,false);
            ic = 1;
            for i=1:numel(Input)
                runs = Input{i};
                for j=1:numel(runs)
                    runs(j).ixexper_id= ic;
                    ic = ic+1;
                end
                Input{i} = runs;
            end
            Input{2}(1)= Input{1}(1);
            Input{3}(1)= Input{1}(1);

            [result,skipped_inputs,this_ixexperid_map,renum_map] = IX_experiment.combine(Input,true);

            hash_defined = arrayfun(@(x)(x.hash_defined),result);
            assertTrue(all(hash_defined));
            assertTrue(isempty(renum_map));


            cai = [Input{1},Input{2}(2:10),Input{3}(2:10)];
            assertEqual(cai,result);
            assertTrue(iscell(skipped_inputs))
            assertEqual(numel(skipped_inputs),3);
            sis = false(1,10);
            assertEqual(skipped_inputs{1},sis); % all included
            sis(1) = true;
            assertEqual(skipped_inputs{2},sis); % first skipped
            assertEqual(skipped_inputs{3},sis); % first skipped

            keys = this_ixexperid_map.keys();
            for i=1:numel(keys)
                id = this_ixexperid_map.get(keys(i));
                assertEqual(uint32(result(id).ixexper_id),keys(i));
            end
        end

        function test_combine_multirun_same_headers_legacy_works(~)
            Input = test_IX_experiment.build_IX_array_blocks(10,3);
            Input{2}(1)= Input{1}(1);
            Input{3}(1)= Input{1}(1);

            [result,skipped_inputs,this_ixexperid_map,renum_map] = IX_experiment.combine(Input,true);

            hash_defined = arrayfun(@(x)(x.hash_defined),result);
            assertTrue(all(hash_defined));
            assertTrue(isempty(renum_map));


            cai = [Input{1},Input{2}(2:10),Input{3}(2:10)];
            assertEqual(cai,result);
            assertTrue(iscell(skipped_inputs))
            assertEqual(numel(skipped_inputs),3);
            sis = false(1,10);
            assertEqual(skipped_inputs{1},sis); % all included
            sis(1) = true;
            assertEqual(skipped_inputs{2},sis); % first skipped
            assertEqual(skipped_inputs{3},sis); % first skipped

            keys = this_ixexperid_map.keys();
            for i=1:numel(keys)
                id = this_ixexperid_map.get(keys(i));
                assertEqual(uint32(result(id).ixexper_id),keys(i));
            end
        end

        function test_combine_multirun_works(~)
            Input = test_IX_experiment.build_IX_array_blocks(10,3,false);

            [result,skipped_inputs,this_ixexperid_map,renum_map] = IX_experiment.combine(Input);

            hash_defined = arrayfun(@(x)(x.hash_defined),result);
            assertTrue(all(hash_defined));
            assertFalse(isempty(renum_map));

            cai = [Input{:}];
            assertEqual(cai,result);
            assertTrue(iscell(skipped_inputs))
            assertEqual(numel(skipped_inputs),3);
            skipped_inputs = [skipped_inputs{:}];
            assertEqual(numel(skipped_inputs),30);
            assertFalse(any(skipped_inputs)); % nothing skipped

            keys = this_ixexperid_map.keys();
            for i=1:numel(keys)
                id = this_ixexperid_map.get(keys(i));
                assertEqual(uint32(result(id).ixexper_id),keys(i));
            end
            % The pixels from other from these runs have to be renumerated
            % to maintain consistency of sqw object
            id_num = 1:10;
            for i=1:numel(renum_map)
                id_val = (i-1)*10+id_num;
                assertEqual(renum_map{i},[id_num;id_val]);
            end

        end

        function test_combine_multirun_legacy_works(~)
            Input = test_IX_experiment.build_IX_array_blocks(10,3,true);

            [result,skipped_inputs,this_ixexperid_map,renum_map] = IX_experiment.combine(Input);

            hash_defined = arrayfun(@(x)(x.hash_defined),result);
            assertTrue(all(hash_defined));
            assertTrue(isempty(renum_map));

            cai = [Input{:}];
            assertEqual(cai,result);
            assertTrue(iscell(skipped_inputs))
            assertEqual(numel(skipped_inputs),3);
            skipped_inputs = [skipped_inputs{:}];
            assertEqual(numel(skipped_inputs),30);
            assertFalse(any(skipped_inputs)); % nothing skipped

            keys = this_ixexperid_map.keys();
            for i=1:numel(keys)
                id = this_ixexperid_map.get(keys(i));
                assertEqual(uint32(result(id).ixexper_id),keys(i));
            end
        end

        function test_combine_single_runs_eq_headers(~)
            data = test_IX_experiment.build_IX_array(10);
            data(2) = data(7);
            Input = num2cell(data);

            [result,skipped_inputs,this_ixexperid_map] = data.combine(Input,true);
            hash_defined = arrayfun(@(x)(x.hash_defined),result);
            assertTrue(all(hash_defined));


            assertEqual([data(1:6),data(8:10)],result);
            assertTrue(iscell(skipped_inputs))
            skipped_inputs = [skipped_inputs{:}];
            assertEqual(numel(skipped_inputs),10);
            assertTrue(skipped_inputs(7)); % 7th skipped
            assertFalse(all(skipped_inputs(1:5))); % left itact
            assertFalse(all(skipped_inputs(7:9))); % left itact

            keys = this_ixexperid_map.keys();
            for i=1:numel(keys)
                id = this_ixexperid_map.get(keys(i));
                assertEqual(result(id).ixexper_id,double(keys(i)));
            end
        end

        function test_combine_one(~)
            data = test_IX_experiment.build_IX_array(10);
            [result,skipped_inputs,this_ixexperid_map,subst_map] = data.combine({data(1)},true);

            assertEqual(data(1),result);
            assertEqual(skipped_inputs,{false});
            assertTrue(isempty(subst_map));
            rmd = result.get_experid_map();
            assertEqual(rmd.keys,this_ixexperid_map.keys);
            assertEqual(rmd.values,this_ixexperid_map.values);
        end

        function test_combine_empty_throw(~)
            data = test_IX_experiment.build_IX_array(10);
            assertExceptionThrown(@()data.combine({},true,true), ...
                'HERBERT:IX_experiment:invalid_argument');
        end

        function test_combine_single_runs_differ_ds_forces_renum(~)
            data = test_IX_experiment.build_IX_array(10,false);
            data(7).ixexper_id = 2; % this ds now looks like coming from
            % different sqw as has ixexper_id equal to ixexper_id from
            % data(2)
            Input = num2cell(data);

            [exper,skipped_inputs,this_ixexperid_map,subst_map] = IX_experiment.combine(Input);
            assertFalse(any([skipped_inputs{:}]));
            assertTrue(this_ixexperid_map.trivial_map);
            assertEqual(data,exper) % ixeper_id value do not participate in comparison
            sm = [subst_map{:}];
            assertEqual(sm(:,7),[2;7]);
            sm(1,7) = 7;
            assertEqual(sm,[1:10;1:10]);
        end

        function test_combine_single_runs_throws_on_emode(~)
            data = test_IX_experiment.build_IX_array(10);
            data(2).emode = 2;
            Input = num2cell(data);
            assertExceptionThrown(@()IX_experiment.combine(Input), ...
                'HORACE:IX_experiment:not_implemented');
        end

        function test_combine_single_runs_throws_on_same(~)
            data = test_IX_experiment.build_IX_array(10);
            data(2) = data(7);
            for i=1:numel(data)
                data(i).ixexper_id = i;
            end
            Input = num2cell(data);
            assertExceptionThrown(@()IX_experiment.combine(Input), ...
                'HORACE:IX_experiment:invalid_argument');
        end

        function test_combine_single_runs_works(~)
            % generate modern run_id(s)
            data = test_IX_experiment.build_IX_array(10,false);
            Input = num2cell(data);

            [result,skipped_inputs,this_ixexperid_map,subst_map] = IX_experiment.combine(Input);
            hash_defined = arrayfun(@(x)(x.hash_defined),result);
            assertTrue(all(hash_defined));
            assertTrue(isempty(subst_map));


            assertEqual(data,result);
            assertTrue(iscell(skipped_inputs))
            skipped_inputs = [skipped_inputs{:}];
            assertEqual(numel(skipped_inputs),10);
            assertFalse(any(skipped_inputs)); % nothing skipped

            keys = this_ixexperid_map.keys();
            for i=1:numel(keys)
                id = this_ixexperid_map.get(keys(i));
                assertEqual(result(id).ixexper_id,id);
            end
        end

        function test_combine_single_legacy_runs_works(~)
            % generate legacy run-id
            data = test_IX_experiment.build_IX_array(10,true);
            Input = num2cell(data);

            [result,skipped_inputs,this_ixexperid_map,subst_map] = IX_experiment.combine(Input);
            hash_defined = arrayfun(@(x)(x.hash_defined),result);
            assertTrue(all(hash_defined));
            assertTrue(isempty(subst_map));


            assertEqual(data,result);
            assertTrue(iscell(skipped_inputs))
            skipped_inputs = [skipped_inputs{:}];
            assertEqual(numel(skipped_inputs),10);
            assertFalse(any(skipped_inputs)); % nothing skipped

            keys = this_ixexperid_map.keys();
            for i=1:numel(keys)
                id = this_ixexperid_map.get(keys(i));
                assertEqual(uint32(result(id).run_id),keys(i));
                assertEqual(result(id).ixexper_id,double(keys(i)));
            end
        end

        %==================================================================
        function test_comparison_hash_neq(~)
            exp1 = IX_experiment('my_file','my_path',1,20,1,'psi',10);
            exp2 = exp1;
            exp2.omega = 4;

            ch1 = exp1.build_hash();
            ch2 = exp2.build_hash();
            assertFalse(isequal(ch1,ch2));
        end

        function test_goniometer_key_construction(~)
            gon = Goniometer(10,[0,1,0],[1,0,0]);

            exp1 = IX_experiment('my_file','my_path',666,10,1,1:9, ...
                'goniometer',gon);

            assertEqual(exp1.psi,10)
            assertEqual(exp1.filename,'my_file')
            assertEqual(exp1.cu,[0,1,0])
            assertEqual(exp1.cv,[1,0,0])
            % in all practical cases offset here is 0. Leave offset field
            % but do not use it for all practical purposes
            assertEqual(exp1.uoffset,[0,0,0,0])
        end

        function test_goniometer_construction(~)
            gon = Goniometer(10,[0,1,0],[1,0,0]);
            exp1 = IX_experiment('my_file','my_path',666,10,1,1:9,gon);

            assertEqual(exp1.psi,10)
            assertEqual(exp1.filename,'my_file')
            assertEqual(exp1.cu,[0,1,0])
            assertEqual(exp1.cv,[1,0,0])
        end

        function test_comparison_hash_eq(~)
            exp1 = IX_experiment('my_file','my_path',1,20,1,'psi',10);
            exp2 = exp1;
            exp2.filepath = 'other_path';

            ch1 = exp1.build_hash();
            ch2 = exp2.build_hash();
            assertEqual(ch1,ch2,'-ignore_str');
        end

        function test_convert_from_old_binfile_header_full_data(~)
            in = struct(...
                'filename', 'sqw_T30p0_BG_fourA_newVANA237684.nxspe',...
                'filepath','spe/',...
                'efix',5.1128,...
                'emode',1,...
                'alatt',[23.9549 8.0759 18.2733],...
                'angdeg',[90 90 90.3000],...
                'cu',[0 1 0],...
                'cv',[0 0 1],...
                'psi',-0.2409,...
                'omega',0,...
                'dpsi',0,...
                'gl',0,...
                'gs',0,...
                'en',-1:0.02:4.2,...
                'uoffset', zeros(4,1),...
                'u_to_rlu',[3.8125,-0.0200,0.0,0;0,1.2853,0.,0;0,0,2.9083,0;0,0,0,1],...
                'ulen',[1 1 1 1],...
                'ulabel','');
            in.ulabel = {'Q_\zeta'  'Q_\xi'  'Q_\eta'  'E'};

            [exp_rec,alatt,angdeg] = IX_experiment.build_from_binfile_header(in);

            assertEqual(alatt,in.alatt);
            assertEqual(angdeg,in.angdeg);
            assertEqual(exp_rec.angular_units,'rad');
            assertEqual(exp_rec.run_id,30);  % this is wrong, but this is
            % old file, so was build with this kind of processing and
            % should properly process it
            assertEqual(exp_rec.psi,in.psi);
            assertTrue(isempty(exp_rec.u_to_rlu));
        end

        function test_convert_to_and_from_old_binfile_headers(~)
            exp = IX_experiment();
            exp(1).filename = 'aa';
            exp(1).filepath = 'bc';
            exp.run_id = 10;
            exp.en = 1:10;
            exp.psi = 10;


            oh = exp.convert_to_binfile_header('-alatt_angdeg',[1,2,3],[90,90,90]);
            oh.filename = 'aa';

            [exp_rec,alatt,angdeg] = IX_experiment.build_from_binfile_header(oh);

            assertEqual(alatt,[1,2,3]);
            assertEqual(angdeg,[90,90,90]);
            exp.run_id = NaN;
            % old headers are stored in radians
            exp.angular_units = 'rad';
            assertEqual(exp,exp_rec,'-nan_equal');
        end

        function test_get_runids(~)
            exp = [IX_experiment(),IX_experiment()];
            exp(1).filename = 'aa';
            exp(1).filepath = 'bc';
            exp(1).run_id = 10;
            exp(2).filename = 'bb';
            exp(2).filepath = 'de';
            exp(2).run_id = 20;

            ids = exp.get_ixexper_ids();
            assertEqual(ids,[10,20]);

        end

        function test_convert_to_and_from_binfile_headers_empty_fn(~)
            exp = IX_experiment();
            exp(1).filename = '';
            exp(1).filepath = 'bc';
            exp.run_id = 10;
            exp.en = 1:10;
            exp.psi = 10;

            oh = exp.convert_to_binfile_header('-alatt_angdeg',[1,2,3],[90,90,90]);

            [exp_rec,alatt,angdeg] = IX_experiment.build_from_binfile_header(oh);

            assertEqual(alatt,[1,2,3]);
            assertEqual(angdeg,[90,90,90]);
            % old headers are stored in radians
            exp.angular_units = 'rad';

            assertEqual(exp,exp_rec);
        end

        function test_convert_to_and_from_binfile_headers(~)
            exp = IX_experiment();
            exp(1).filename = 'aa';
            exp(1).filepath = 'bc';
            exp.run_id = 10;
            exp.en = 1:10;
            exp.psi = 10;

            oh = exp.convert_to_binfile_header('-alatt_angdeg',[1,2,3],[90,90,90]);

            [exp_rec,alatt,angdeg] = IX_experiment.build_from_binfile_header(oh);

            assertEqual(alatt,[1,2,3]);
            assertEqual(angdeg,[90,90,90]);
            % old headers are stored in radians
            exp.angular_units = 'rad';
            assertEqual(exp,exp_rec);

        end

        function test_recover_from_v1_structure_array(~)
            clWarn = set_temporary_warning('off','HORACE:IX_experiment:undefined_run_id');

            exp = [IX_experiment(),IX_experiment()];
            exp(1).filename = 'aa';
            exp(1).filepath = 'bc';
            exp(2).filename = 'bb';
            exp(2).filepath = 'de';


            v1_struct = exp.to_struct(); % this is v2 structure
            % prepare v1 structure, not to bother with the file storage
            v1_struct.version = 1;
            v1_struct.array_dat = rmfield(v1_struct.array_dat,'run_id');

            exp_rec = hashable.from_struct(v1_struct);
            % old IX_experiment structures in all practical cases were storing
            % angular units in radian, so we restoring old versions as
            % radians
            for i=1:numel(exp)
                exp(i).angular_is_degree = false;
                % This is probably not a good idea on IX_experiment level,
                % as this number defines also pixel id so at sqw level relation
                % may be broken. But at IX_exper-- that what is reasonable
                % let's do this for the time being
                exp(i).ixexper_id = i;
                assertEqual(exp_rec(i).ixexper_id,i)
            end

            assertEqual(exp,exp_rec,'-nan_equal');
        end

        function test_recover_from_v1_structure_single(~)
            exp = IX_experiment();
            exp.filename = 'aa';
            exp.filepath = 'bc';
            clWarn = set_temporary_warning('off','HORACE:IX_experiment:undefined_run_id');

            v1_struct = exp.to_struct(); % this is v2 structure
            % prepare v1 structure, not to bother with the file storage
            v1_struct.version = 1;
            v1_struct = rmfield(v1_struct,'run_id');

            exp_rec = serializable.from_struct(v1_struct);

            % old IX_experiment structures in all practical cases were storing
            % angular units in radian, so we restoring old versions in
            % radians
            exp.angular_is_degree = false;


            assertEqual(exp,exp_rec,'-nan_equal');
        end

        function test_set_invalid_runid_throws(~)
            exp = IX_experiment();
            function setter(obj,val)
                obj.run_id = val;
            end

            assertExceptionThrown(@()setter(exp,'a'),...
                'HERBERT:IX_experiment:invalid_argument');

            assertExceptionThrown(@()setter(exp,[1,2]),...
                'HERBERT:IX_experiment:invalid_argument');
        end

        function test_set_invalid_ixeper_num_throws(~)
            exp = IX_experiment();
            function setter(obj,val)
                obj.ixexper_id = val;
            end

            assertExceptionThrown(@()setter(exp,'a'),...
                'HERBERT:IX_experiment:invalid_argument');

            assertExceptionThrown(@()setter(exp,[1,2]),...
                'HERBERT:IX_experiment:invalid_argument');
        end

        function test_set_get_single_runid(~)
            exp = IX_experiment();
            assertTrue(isnan(exp.run_id))
            assertTrue(isnan(exp.ixexper_id))
            exp.run_id = 10;
            assertEqual(exp.run_id,10);
            assertEqual(exp.ixexper_id,10)

            exp.run_id = NaN;
            assertTrue(isnan(exp.run_id))
            assertTrue(isnan(exp.ixexper_id))
        end

        function test_full_construnctor(~)
            par_names={'filename', 'filepath','run_id', 'efix','emode','en','psi','cu',...
                'cv','omega','dpsi','gl','gs','angular_units'};
            par_val = {'my_file','my_name',666,10,1,[1,2,4,8]',10,[1,0,0],[0,1,0],...
                1,2,3,4,'rad'};
            angular_val = {'psi','omega','dpsi','gl','gs'};
            % For debugging: Construction fields are defined as u,v
            %exp0 = IX_experiment();
            %assertEqual(par_names',exp0.constructionFields());
            pv_map = containers.Map(par_names,par_val);

            exp = IX_experiment(par_val{:});

            fn = exp.constructionFields();
            for i=1:numel(fn)
                prop_name = fn{i};
                if ismember(prop_name,angular_val)
                    expected_val = deg2rad(pv_map(prop_name));
                else
                    if strcmp(prop_name,'v')
                        expected_val = pv_map('cv');
                    elseif strcmp(prop_name,'u')
                        expected_val = pv_map('cu');
                    else
                        expected_val = pv_map(prop_name);
                    end
                end
                assertEqual(exp.(prop_name),expected_val, ...
                    sprintf('invalid value "%s" for field "%s"', ...
                    disp2str(exp.(prop_name)),fn{i}));
            end
        end
    end
end
