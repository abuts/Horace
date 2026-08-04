classdef IX_experiment < Goniometer
    %IX_EXPERIMENT -- transient class which describes transformation of a
    %single run into Crystal Cartesian coordinate system during sqw file
    %generation
    %
    % may be replaced by instr_proj in a future or may be should build
    % close ties with such projection.
    %
    % NOTE:
    % Two IX_experiments with the same goniometer, energy+mode and short
    % filename but with different run_id are considered equal. This is
    % because hashableFields for IX_experiment do not include rin_id fields
    % so hash is not affected by its value.
    %
    % Run-id is notionally related to real experimental run, but actually
    % have meaning of a tag, which connects particular IX_experiment with
    % particular pixel (neutron event) through This is the logical connection,
    % build at sqw generation and maintained during operations with sqw
    % object
    %
    % Run-id connection with actual experimental run is useful but
    % not-mandatory feature.
    properties(Dependent)
        filename; % name of the file which was the source of data for this
        %         % experiment
        filepath; % path where the experiment data were initially stored
        run_id;   % The number, which uniquely defines the run where
        %         % the experiment data came from. Very often, the number
        %         % of run on an instrument
        ixexper_id; % The number, which uiquely identifies this dataset.
        %         % as part of all datasets or runs contributed into the
        %         % sqw object these data came from.
        %         % For resolution convolution to work correctly,
        %         % this identifier also must be stored within the PixelData,
        %         % providing connection between the particular pixel and
        %         % the particuar experiment info, pointing to appropriate
        %         IX_experiment
        emode;
        efix;
        en;  % array of all energy transfers, present in the experiment

    end
    properties(Dependent,Hidden)
        % returns goniometer sliced from this object
        goniometer;
        % redundant property. Was inv(b_matrix). left for compatibility
        % with legacy alignment, as it multiplies it by alignment rotation
        % matrix and keeps legacy alignment matrix this way.
        u_to_rlu;
    end
    properties(Hidden)
        attached_instr_hash = '' % helper property, used in comparison of
        % different IX experiments. If defined (not empty) used in
        % comparison. Not used in IX_experiment hashes. Used only in
        % combine algorithms to identify unique IX_experiments and pixels
        % corresponding to them.
        
        % these properties are not used in Horace-4 but left for compatibility
        % with Horace-3 file format when it read/updated from/to Horace-3
        % format files.
        ulabel = {'','','',''};
        ulen = [1,1,1,1];
    end

    properties(Hidden)
        % Never usefully used except loading from old files so candidate
        % for removal
        uoffset=[0,0,0,0];  % Always 0.
    end
    properties(Access=protected)
        filename_=''
        filepath_='';
        run_id_ = NaN;
        ixexper_id_ = [];
        emode_ = 0;
        en_ = zeros(0,1);
        efix_ = 0;
        u_to_rlu_ = [];
    end
    methods
        function obj = IX_experiment(varargin)
            % IX_EXPERIMENT Construct an instance of this class
            obj.angular_units = 'deg';
            if nargin==0
                return
            end
            obj = obj.init(varargin{:});
        end

        function obj = init(obj,varargin)
            % construct non-empty instance of this class
            % Usage:
            %   obj = init(obj,filename, filepath, efix,emode,cu,cv,psi,...
            %               omega,dpsi,gl,gs,en,uoffset,u_to_rlu,ulen,...
            %               ulabel,run_id)
            %
            obj = init_(obj,varargin{:});
        end
        %------------------------------------------------------------------
        % ACCESSORS:
        function fn = get.filename(obj)
            fn = obj.filename_;
        end
        function obj = set.filename(obj,val)
            if ~(ischar(val) || isstring(val))
                error('HERBERT:IX_experiment:invalid_argument',...
                    'filename can be only character array or string. It is %s',...
                    class(val))
            end
            obj.filename_     = val;
            obj = obj.clear_hash();
        end
        %
        function fn = get.filepath(obj)
            fn = obj.filepath_;
        end
        function obj = set.filepath(obj,val)
            if ~(ischar(val) || isstring(val))
                error('HERBERT:IX_experiment:invalid_argument',...
                    'filename can be only character array or string. It is %s',...
                    class(val))
            end
            obj.filepath_ = val;
        end
        %
        function id = get.run_id(obj)
            id =obj.run_id_;
        end
        function obj = set.run_id(obj,val)
            obj = check_and_set_id_prop(obj,val,'run_id_');
        end

        function id = get.ixexper_id(obj)
            if isempty(obj.ixexper_id_) % for compartibility
                id =obj.run_id_;      % with old data containing run_id only.
            else
                id =obj.ixexper_id_;
            end
        end
        function obj = set.ixexper_id(obj,val)
            obj = check_and_set_id_prop(obj,val,'ixexper_id_');
        end

        function ids = get_ixexper_ids(obj)
            % retrieve all run_ids, which may be present in the array of
            % rundata objects.
            n_obj = numel(obj);
            ids = zeros(1,n_obj);
            for in=1:n_obj
                % runID-s obtained from different sources may be int, uint32
                %  or double. To achieve consistensy, let's make them all
                %  double.
                ids(in) = double(obj(in).ixexper_id);
            end
        end
        function idmap = get_experid_map(obj)
            % retrieve all run_ids, which may be present in the array of
            % rundata objects and build run_id map from them. run_id map
            % used for finding particular element's position given its
            % run_id

            n_obj = numel(obj);
            ids = zeros(1,n_obj);
            ind = 1:numel(obj);
            trivial_map = true;
            for ii = ind
                ids(ii) = obj(ii).ixexper_id;
                if ids(ii) ~= ii
                    trivial_map = false;
                end
            end
            idmap = fast_map(ids,ind,trivial_map);
        end
        %
        function mode = get.emode(obj)
            mode = obj.emode_;
        end
        function obj = set.emode(obj,val)
            if ~isnumeric(val) || ~isscalar(val) || val<0 || val>2
                error('HERBERT:IX_experiment:invalid_argument',...
                    'Transformation mode can be numeric scalar in range from 0 to 2.\n It is: %s',...
                    disp2str(val));
            end
            obj.emode_ = val;
            obj = obj.clear_hash();
        end
        %
        function en = get.en(obj)
            en = obj.en_;
        end
        function en = get_en(obj,bin_centers)
            % return energy transfer values with possibility to
            % retrieve bin centers.
            % Return rows, wrt get.en, which always returns columns
            en = obj.en_;
            if bin_centers
                en = 0.5*(en(1:end-1)+en(2:end));
            end
            en = en(:)';
        end
        function obj = set.en(obj,val)
            if ~isnumeric(val)
                error('HERBERT:IX_experiment:invalid_argument',...
                    'energy transfers have to be array of numeric values. It is: %s',...
                    class(val));
            end
            obj.en_ = val(:);
            obj = obj.clear_hash();
        end
        % method which usually used on array of IX_experiments to obtain
        % array of energy transfers and additional information (uniqueness
        % of energy transfers). Works on single IX_experiment too.
        [en_transf,unique_idx]  = get_en_transfer(obj,bin_centre,get_lidx)
        %
        function ef = get.efix(obj)
            ef = obj.efix_;
        end
        function obj = set.efix(obj,val)
            if val < 0
                error('HERBERT:IX_experiment:invalid_argument',...
                    'efix (incident energy) can not be negative')
            end
            obj.efix_ = val;
            obj = obj.clear_hash();
        end
        %
        function mat = get.u_to_rlu(obj)
            if isempty(obj.u_to_rlu_)
                mat = [];
            else
                mat = eye(4);
                mat(1:3,1:3) = obj.u_to_rlu_;
            end
        end
        function obj = set.u_to_rlu(obj,val)
            if isempty(val)
                obj.u_to_rlu_ = [];
                return
            end
            if all(size(val)== [4,4])
                val = val(1:3,1:3);
            end
            if ~all(size(val) == [3,3])
                error('HERBERT:IX_experiment:invalid_argument',...
                    'input u_to_rlu matrix have to have size 3x3 or 4x4')
            end
            if all(abs(subdiag_elements(val))<4.e-7)
                val = [];
            end
            obj.u_to_rlu_ = val;
        end
        %
        function gon = get.goniometer(obj)
            str = obj.to_bare_struct();
            str.u = obj.cu;
            str.v = obj.cv;
            gon = Goniometer(str);
        end
        function obj = set.goniometer(obj,val)
            if isstruct(val)
                if isfield(val,'cu')
                    val.u = val.cu;
                end
                if isfield(val,'cv')
                    val.v = val.cv;
                end
            elseif isa(val,'Goniometer')
                val = val.to_bare_struct();
            else
                error('HORACE:IX_experiment:invalid_argument', ...
                    'Goniometer property accepts input as a class "Goniometer" or a structure, convertible into Goniometer.\n Provided %s', ...
                    class(val));
            end
            obj = obj.from_bare_struct(val);
            obj = obj.clear_hash();
        end
    end
    %----------------------------------------------------------------------
    methods
        % SQW_binfile_common methods related to saving to binfile and
        % run_id scrambling:
        function old_hdr = convert_to_binfile_header(obj,mode,arg1,arg2,nomangle)
            % convert to the header structure, to be stored in the old
            % binary files.
            %
            % Inputs:
            % Required:
            % obj   -- the experiment data header object to convert -
            % mode  --
            %    = '-inst_samp' : the next 2 arguments are an instrument
            %                     and sample respectively
            %    = '-alatt_angdeg' : the next 2 arguments are the alatt and
            %                        angdeg values of the run respectively.
            %                        In this case a null instrument and
            %                        sample with these values are created
            %                        and used.
            % arg1  --
            %    = instrument to set if mode == '-inst_samp'
            %    = alatt -- lattice cell sizes (3x1 vector) if mode ==
            %               '-alatt_angdeg'
            % arg2  --
            %    = sample to set if node == '-inst_samp'
            %    = angdeg --lattice angles (3x1 vector) if mode ==
            %               '-alatt_angdeg'
            % Optional:
            % nomangle -- if false or absent, mangle (append to the end)
            %             file name with run_id (if one is defined)
            %
            % Outputs:
            % old_hdr  -- struct with the old-style header data
            %
            if ~exist('nomangle','var')
                nomangle = false;
            end
            old_hdr = convert_to_binfile_header_(obj,mode,arg1,arg2,nomangle);
        end
        %
        %
        %
    end
    methods(Static)
        function [obj,file_id_array,skipped_inputs,this_runid_map] = ...
                combine(exper_cellarray,allow_eq_headers,varargin)
            % method combines input IX_experiment array(s) with elements
            % contained in exper_cellarray, identifying possible duplicates
            % and either ignoring them, or throwing error depending on
            % input parameters.
            %
            % Inputs:
            % exper_cellarray -- cellarray containing IX_experiments arrays
            %                    or Experiment classes to combine their
            %                    IX_experiments into obj.
            % allow_eq_headers-- if true, headers with the same runid and
            %                    same values are allowed and accounted for
            %                    in combine operations. If false, routine
            %                    throws HORACE:IX_experiment:invalid_argument
            %                    if the IX_experiment have the same run_id
            %                    and values.
            % Optional:
            % runid_map       -- the map containing information about
            %                    run_id(s) stored in the object as keys
            %                    and pointing to the number of element in
            %                    obj array as the value.
            %
            % Returns:
            % obj             -- resulting array, containing unique
            %                    instances of IX_experiment classes with
            %                    all non-unique IX_experiments excluded.
            % skipped_inputs  -- cellarray of logical arrays containing true where
            %                    input object was dropped and false where it has been
            %                    kept
            % file_id_array   -- array contains run_ids for each input
            %                    IX_experiment value present in exper_cellarray.
            %                    Where input IX_experiments with equal run_id-s
            %                    and values are rejected, corresponding
            %                    elements of this array contain the
            %                    values of rejected run_id-s. These values
            %                    will be used  in calculations of pixels
            %                    run_id for each contributing file.
            % this_runid_map --  the map which connects run_id(s) of data,
            %                    stored in the obj as keys, with the
            %                    positions of the data objects in the
            %                    object array as values.
            if nargin < 3
                allow_eq_headers = false;
            end
            [obj,file_id_array,skipped_inputs,this_runid_map] = combine_(...
                exper_cellarray,allow_eq_headers,varargin{:});
        end

        %------------------------------------------------------------------
        % SQW_binfile_common methods related to saving to old format binfile and
        % run_id scrambling:
        function [obj,alatt,angdeg] = build_from_binfile_header(inputs)
            % Inputs: the old header structure, stored in binfile
            old_fldnms = {'filename','filepath','efix','emode','en','cu',...
                'cv','psi','omega','dpsi','gl','gs','uoffset','u_to_rlu'};
            obj = IX_experiment();
            % disable check of interdependent properties until all are set
            obj.do_check_combo_arg = false;
            for i=1:numel(old_fldnms)
                obj.(old_fldnms{i}) = inputs.(old_fldnms{i});
            end
            % enable check for interdependent properties compartibility
            obj.do_check_combo_arg = true;
            % check interdependent properties compartibility and
            % calculate all caches if necessary.
            obj = obj.check_combo_arg();
            % old headers always contain angular values in radians
            obj.angular_is_degree_ = false;
            alatt = inputs.alatt;
            angdeg = inputs.angdeg;
            [runid,filename] = a_loader.extract_id_from_filename(inputs.filename);
            if ~isnan(runid)
                obj.run_id = runid;
                obj.filename = filename;
            end
            if all(abs(subdiag_elements(obj.u_to_rlu))<4*eps('single'))
                obj.u_to_rlu = [];
            end
        end
    end
    %----------------------------------------------------------------------
    % SERIALIZABLE interface
    properties(Constant,Access=private)
        % fields, which fully define IX_experiment part of the public
        % interface to the class
        fields_to_save_ = {'filename','filepath','run_id','ixexper_id','efix','emode','en'};
    end
    methods
        function flds = saveableFields(obj)
            base= saveableFields@Goniometer(obj);
            flds = [IX_experiment.fields_to_save_(:);base(:)];
            if ~isempty(obj(1).u_to_rlu_) || isnan(obj(1).run_id_) % run_id_ is NaN on non-initialized file
                flds = [flds(:);'u_to_rlu'];
            end
        end
        function flds = constructionFields(obj)
            base= constructionFields@Goniometer(obj);
            flds = [IX_experiment.fields_to_save_(1:3)';IX_experiment.fields_to_save_(5:end)';base(:)];
        end
        function flds = hashableFields(~)
            % run_id connects pixels with headers in experiment data.
            % We allow two IX_experiments with the same run-id to be equal
            % Also two experiments with the same filename but different
            % filepath are the same
            %
            % the list of properties which define IX_experiment uniqueness
            % if hashes, build on these properties values are the same,
            % IX_experiments are considered the same
            flds= {'filename','cu','cv','run_id','efix',...
                'psi', 'omega', 'dpsi', 'gl', 'gs'};
        end

        function ver  = classVersion(~)
            % return the version of the IX-experiment class
            ver = 4;
        end
        function obj = check_combo_arg(obj)
            % verify interdependent variables and the validity of the
            % obtained lattice object
            obj = check_combo_arg@Goniometer(obj);
            obj = obj.clear_hash();
            if isscalar(obj.efix_) && obj.efix_ == 0 && obj.emode_ ~=0
                error('HERBERT:IX_experiment:invalid_argument',...
                    'efix (incident energy) can be 0 in elastic mode only. Emode=%d', ...
                    obj.emode_)

            end
        end
    end
    methods(Access=protected)
        function obj = from_old_struct (obj, S)
            obj = from_old_struct@serializable(obj,S);
            if S.version == 1 && numel(obj)>1
                for i=1:numel(obj)
                    obj(i).ixexper_id= i;
                end
            end
        end

        function [S,obj] = convert_old_struct (obj, S, ver)
            % Update structure created from earlier class versions to the current
            % version. Converts the bare structure for a scalar instance of an object.
            % Overload this method for customised

            if ver == 1
                % version 1 does not contain run_id so it is set to NaN
                % and recalculated on sqw object level
                S.run_id = NaN;

            end
            if ver < 4
                if isnan(S.run_id)
                    S.ixexper_id = 1;
                    warning('HORACE:IX_experiment:undefined_run_id', ...
                        ['IX_experiment stored on disk does not have correct run-id(s)\n' ...
                        'It is likely that relationship between IX_experiment and pixels are broken and resolution convolution does not work correctly\n' ...
                        'Check https://pace-neutrons.github.io/Horace/unstable/manual/Data_diagnostics.html#instrument-view-cut\n' ...
                        'on how to diagnose this issue.'])
                else
                    S.ixexper_id = S.run_id;
                end
            end
            % version 3 does not save/load u_to_rlu, ulen, ulabel
            % These fields are redundant for instr_proj and moved
            % to sqw.data (DnD object)

            % Old IX_experiment data were containing angular values in
            % radians
            if isfield(S,'goniometer')
                S.goniometer.angular_is_degree = false;
                obj.goniometer = S.goniometer;
            else
                S.angular_is_degree = false;
            end
            if isfield(S,'cu')
                S.u = S.cu;
            end
            if isfield(S,'cv')
                S.v = S.cv;
            end

            if isfield(S,'u_to_rlu')
                % support for legacy alignment:
                if ~any(subdiag_elements(S.u_to_rlu)>4*eps('single'))
                    S = rmfield(S,'u_to_rlu');
                else
                    obj.u_to_rlu = S.u_to_rlu;
                end
            end
        end
    end

    methods(Static)
        function obj = loadobj(S)
            % crafted loadobj method, calling generic method of
            % saveable class, necessary to load data from old structures
            % + support for legacy alignment matrix
            obj = IX_experiment();
            obj = loadobj@serializable(S,obj);
        end
    end
    methods(Access=private)
        function obj = check_and_set_id_prop(obj,val,name)
            % helper setter used in setting run_id or ixexper_id
            if ~isnumeric(val) || ~isscalar(val)
                error('HERBERT:IX_experiment:invalid_argument',...
                    'property %s can have only single numeric value', ...
                    name(1:end-1));
            end
            obj.(name) = val;
        end

    end
end
