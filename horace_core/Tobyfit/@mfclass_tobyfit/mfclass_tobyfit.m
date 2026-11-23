classdef mfclass_tobyfit < mfclass
    % Simultaneously fits resolution broadened S(Q,w) models to several sqw
    % objects, with optional background functions. The foreground function(s)
    % and background function(s) can be set to apply globally to all datasets,
    % or locally, one function per dataset.
    %
    %
    % mfclass_tobyfit Methods:
    % --------------------------------------
    % To set data:
    %   set_data     - Set data, clearing any existing datasets
    %   append_data  - Append further datasets to the current set of datasets
    %   remove_data  - Remove one or more dataset(s)
    %   replace_data - Replace one or more dataset(s)
    %
    % To mask data points:
    %   set_mask     - Mask data points
    %   add_mask     - Mask additional data points
    %   clear_mask   - Clear masking for one or more dataset(s)
    %
    % To set fitting functions:
    %   set_fun      - Set foreground fit functions
    %   clear_fun    - Clear one or more foreground fit functions
    %
    %   set_bfun     - Set background fit functions
    %   clear_bfun   - Clear one or more background fit functions
    %
    % To set initial function parameter values:
    %   set_pin      - Set foreground fit function parameters
    %   clear_pin    - Clear parameters for one or more foreground fit functions
    %
    %   set_bpin     - Set background fit function parameters
    %   clear_bpin   - Clear parameters for one or more background fit functions
    %
    % To set which parameters are fixed or free:
    %   set_free     - Set free or fix foreground function parameters
    %   clear_free   - Clear all foreground parameters to be free for one or more data sets
    %
    %   set_bfree    - Set free or fix background function parameters
    %   clear_bfree  - Clear all background parameters to be free for one or more data sets
    %
    % To bind parameters:
    %   set_bind     - Bind foreground parameter values in fixed ratios
    %   add_bind     - Add further foreground function bindings
    %   clear_bind   - Clear parameter bindings for one or more foreground functions
    %
    %   set_bbind    - Bind background parameter values in fixed ratios
    %   add_bbind    - Add further background function bindings
    %   clear_bbind  - Clear parameter bindings for one or more background functions
    %
    % To set functions as operating globally or local to a single dataset
    %   set_global_foreground - Specify that there will be a global foreground fit function
    %   set_local_foreground  - Specify that there will be local foreground fit function(s)
    %
    %   set_global_background - Specify that there will be a global background fit function
    %   set_local_background  - Specify that there will be local background fit function(s)
    %
    % To fit or simulate:
    %   fit          - Fit data
    %   simulate     - Simulate datasets at the initial parameter values
    %
    % Fit control parameters and other options:
    %   set_options  - Set options
    %   get_options  - Get values of one or more specific options
    %
    % In addition, specifically for Tobyfit / fitter control:
    %   set_mc_contributions    - Alter which components contribute to the resolution
    %   set_mc_points           - Set the number of Monte Carlo points per pixel
    %   set_resolution_model    - Set replaceable resolution model
    %   set_fitter              - Set fitting algorithm (subclass of mf_fitter_base)
    %   set_fitting_ranges      - Set parameter bounds [lb, ub] for bounded fitters
    %   set_fitter_parameters   - Set additional optimizer parameters
    %   set_refine_crystal      - Refine crystal lattice parameters and orientation
    %   set_refine_moderator    - Refine moderator parameters
    %
    %
    % mfclass_tobyfit Properties:
    % --------------------------------------
    % Data to be fitted:
    %   data         - datasets to be fitted or simulated
    %   mask         - mask arrays to remove data points from fitting or simulation
    %
    % Fit functions:
    %   fun          - foreground fit function handles
    %   pin          - foreground function parameter values
    %   free         - the foreground function parameters that can vary in a fit
    %   bind         - binding of foreground parameters to free parameters
    %
    %   bfun         - foreground fit function handles
    %   bpin         - foreground function parameter values
    %   bfree        - the foreground function parameters that can vary in a fit
    %   bbind        - binding of foreground parameters to free parameters
    %
    % To set functions as operating globally or local to a single dataset
    %   global_foreground - true if a global foreground fit function
    %   local_foreground  - true if a local foreground fit functions
    %   global_background - true if a global background fit function
    %   local_background  - true if a local background fit function(s)
    %
    % Options:
    %   options      - options defining fit control parameters
    %   fitter       - active optimization algorithm (subclass of mf_fitter_base)
    %   fitting_ranges    - parameter bounds [lb, ub] for bounded optimizers
    %   fitter_parameters - additional optimizer parameters
    %
    % In addition, specifically for Tobyfit:
    %   mc_contributions    - Defines which components contribute to the resolution
    %   mc_points           - The number of Monte Carlo points per pixel
    %   refine_crystal      - Crystal orientation refinement parameters
    %   refine_moderator    - Moderator parameter refinement parameters

    % -----------------------------------------------------------------------------
    % <#doc_def:>
    %   mfclass_doc = fullfile(fileparts(which('mfclass')),'_docify')
    %   mfclass_tobyfit_doc = fullfile(fileparts(which('mfclass_tobyfit')),'_docify')
    %
    %   mfclass_tobyfit_doc_purpose_summary_file = fullfile(mfclass_tobyfit_doc,'doc_purpose_summary.m')
    %   mfclass_doc_methods_summary_file = fullfile(mfclass_doc,'doc_methods_summary.m')
    %   mfclass_tobyfit_doc_methods_summary_file = fullfile(mfclass_tobyfit_doc,'doc_methods_summary.m')
    %   mfclass_doc_properties_summary_file = fullfile(mfclass_doc,'doc_properties_summary.m')
    %   mfclass_tobyfit_doc_properties_summary_file = fullfile(mfclass_tobyfit_doc,'doc_properties_summary.m')
    %
    %   class_name = 'mfclass_tobyfit'
    %
    % -----------------------------------------------------------------------------
    % <#doc_beg:> multifit
    %   <#file:> <mfclass_tobyfit_doc_purpose_summary_file>
    %
    %
    % <class_name> Methods:
    % --------------------------------------
    %   <#file:> <mfclass_doc_methods_summary_file>
    %
    %   <#file:> <mfclass_tobyfit_doc_methods_summary_file>
    %
    %
    % <class_name> Properties:
    % --------------------------------------
    %   <#file:> <mfclass_doc_properties_summary_file>
    %
    %   <#file:> <mfclass_tobyfit_doc_properties_summary_file>
    % <#doc_end:>
    % -----------------------------------------------------------------------------

    properties (Access=private, Hidden=true)
        res_model_ = [];
        mc_contributions_ = [];
        mc_points_ = [];
        refine_crystal_ = [];
        refine_moderator_ = [];
    end

    properties (Dependent)
        % Define which components of instrument contribute to resolution function model
        mc_contributions

        % The number of Monte Carlo points per pixel
        mc_points

        % Crystal orientation refinement parameters
        % If crystal refinement will not to be performed, contains [];
        % otherwise a structure with parameters:
        %   alatt       Initial lattice parameters
        %   angdeg      Initial lattice angles
        %   rot         Initial rotation vector (rad) (=[0,0,0])
        %   urot        x-axis in r.l.u. (Default: [1,0,0])
        %   vrot        Defines y-axis in r.l.u. (in plane of urot and vrot)
        %               (Default: [0,1,0])
        %   free        Logical row vector (length=9) (0 fixed, 1 free)
        %               (Default: all free)
        %   fix_alatt_ratio     =true if a,b,c are to be bound (Default: false)
        refine_crystal

        % Moderator parameter refinement parameters
        % If moderator refinement will not to be performed, contains [];
        % otherwise a structure with parameters:
        %	pulse_model     Name of moderator pulse shape model
        %   pin             Pulse shape parameters (row vector)
        %   free            Logical row vector of zeros and ones (0 fixed, 1 free)
        refine_moderator

        % Active resolution convolution model (subclass of tobyfit_res_model_base)
        res_model
    end

    methods
        %------------------------------------------------------------------
        % Constructor
        %------------------------------------------------------------------
        function obj = mfclass_tobyfit (varargin)
            [mf_args, res_model_in, fitter_in, ranges_in, fparams_in] = parse_ctor_args_(varargin{:});
            obj@mfclass(mf_args{:});
            if ~isempty(res_model_in)
                obj = obj.set_resolution_model(res_model_in);
            elseif ~isempty(obj.wrapfun.fun_wrap)
                switch func2str(obj.wrapfun.fun_wrap)
                    case 'tobyfit_DGfermi_resconv'
                        obj.res_model_ = tobyfit_DGfermi_res_model();
                    case 'tobyfit_DGdisk_resconv'
                        obj.res_model_ = tobyfit_DGdisk_res_model();
                end
            end
            if ~isempty(fitter_in)
                obj = obj.set_fitter(fitter_in);
            end
            if ~isempty(ranges_in)
                obj = obj.set_fitting_ranges(ranges_in);
            end
            if ~isempty(fparams_in)
                obj = obj.set_fitter_parameters(fparams_in);
            end
            obj = obj.set_mc_contributions;
            obj = obj.set_mc_points;
            obj = obj.set_refine_crystal (false);
            obj = obj.set_refine_moderator (false);
        end

        %------------------------------------------------------------------
        % Set/get methods
        %------------------------------------------------------------------
        function out = get.res_model (obj)
            out = obj.res_model_;
        end
        function out = get.mc_contributions (obj)
            out = obj.mc_contributions_;
        end

        function out = get.mc_points (obj)
            out = obj.mc_points_;
        end

        function out = get.refine_crystal (obj)
            out = obj.refine_crystal_;
        end

        function out = get.refine_moderator (obj)
            out = obj.refine_moderator_;
        end

        %------------------------------------------------------------------
        % Interfaces to extended superclass methods
        %------------------------------------------------------------------
        % Extend set_fun and set_bfun solely to provide tailored documentation

        % Set resolution convolution model
        obj = set_resolution_model(obj, res_model)

        % Set parameter fitting ranges [lb, ub] for bounded optimizers
        obj = set_fitting_ranges(obj, varargin)

        % Set additional fitter parameters
        obj = set_fitter_parameters(obj, varargin)

        % Set foreground function or functions
        obj = set_fun(obj,varargin)

        % Set background function or functions
        obj = set_bfun(obj,varargin)

        % Perform a fit of the data using the current functions and starting parameter values
        [data_out, fitdata, ok, mess, varargout] = fit (obj, varargin)

        % Perform a simulation of the data using the current functions and starting parameter values
        [data_out, calcdata, ok, mess] = simulate (obj, varargin)

    end

    methods (Access=private)
        %------------------------------------------------------------------
        % Methods in the defining folder but which need to be kept private
        %------------------------------------------------------------------
        [ok, mess, obj, xtal] = refine_crystal_pack_parameters_ (obj, xtal_opts)
        [ok, mess, obj, modshape] = refine_moderator_pack_parameters_ (obj, mod_opts)
    end
end
