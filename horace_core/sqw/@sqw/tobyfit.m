function obj = tobyfit (varargin)
% Simultaneously fits resolution broadened S(Q,w) models to sqw objects.
%
% For a direct geometry Fermi chopper instrument:
%   >> myobj = tobyfit (w1, w2, ...)      % w1, w2 sqw objects or arrays of objects
%
% More generally:
%   >> myobj = tobyfit (w1, w2, ..., 'res_model', model, 'fitter', fitter, ...
%                'fitting_ranges', ranges, 'fitter_parameters', params)
%
%   'res_model'   Replaceable resolution model (subclass of tobyfit_res_model_base)
%                 or instrument identifier (e.g. 'IX_inst_DGfermi', 'IX_inst_DGdisk')
%   'fitter'      Replaceable fitting algorithm (subclass of mf_fitter_base)
%                 or fitter name ('lsqr', 'fminsearch', 'lsqnonlin', 'particleswarm')
%   'fitting_ranges'     Parameter lower/upper bounds ([lb, ub]) for bounded fitters
%   'fitter_parameters'  Struct of extra optimizer options (e.g. SwarmSize, MaxIterations)
%
% Legacy syntax (deprecated):
%   >> myobj = tobyfit (w1, w2, ..., instrument_type)
%       where instrument_type was 'fermi' or 'disk'
%
% This creates a fitting object of class mfclass_tobyfit with the provided
% data, which can then be manipulated to add further data, set the fitting
% functions, initial parameter values etc. and fit or simulate the data.
% For details on how to do this <a href="matlab:help('mfclass_tobyfit');">Click here</a>
%
% Tobyfit fits model(s) for S(Q,w) as the foreground function(s), and
% function(s) of the plot axes as the background function(s)
%
% For the format of foreground fit functions see the following examples:
% <a href="matlab:edit('example_sqw_spin_waves');">Damped spin waves</a>
% <a href="matlab:edit('example_sqw_flat_mode');">Dispersionless excitations</a>
%
% The format of the background fit functions depends on the number of plot
% axes for each sqw object. For examples see:
% <a href="matlab:edit('example_1d_function');">example_1d_function</a>
% <a href="matlab:edit('example_2d_function');">example_2d_function</a>
% <a href="matlab:edit('example_3d_function');">example_3d_function</a>


% Original author: T.G.Perring

% ------------------------------------------------------------------------------
% Tobyfit (1 Jan 2018 onwards)
% ----------------------------
% Get resolution function model type

% Determine the instrument model
% There must be at least one sqw object in the argument list or this method
% would not have been called. However, it may not be the leading argument.
% We demand that the leading argument is an sqw object.
% The input could be any number of sqw object arrays, followed by other arguments
% hence the slightly involved procedure

is_sqw_object = cellfun(@(x) isa(x,'sqw'), varargin);
ind = find(~is_sqw_object,1);

if isempty(ind)
    sqws = varargin;
    argi = {};
else
    nsqw = ind-1;
    sqws = varargin(1:nsqw);
    argi = varargin(nsqw+1:end);
end

if isempty(sqws)
    error('HORACE:tobyfit:invalid_argument', 'There must be at least one leading sqw object in the input argument list to Tobyfit')
end

if any(arrayfun(@(x) x.pix.is_filebacked, [sqws{:}]))
    error('HORACE:tobyfit:invalid_argument', ...
        ['Tobyfit does not currently support file-backed sqw objects\n' ...
        ' In order to attempt to run please convert all sqws to memory-backed:\n' ...
        '  >> my_sqw.pix = PixelDataMemory(my_sqw.pix);'])
end


% Parse optional trailing 'res_model' / 'fitter' / range keywords or legacy char flag
res_model_arg = [];
fitter_arg = [];
ranges_arg = [];
fparams_arg = [];
if isscalar(argi) && (ischar(argi{1}) || isstring(argi{1})) && ...
        ~any(strcmpi(char(argi{1}), {'res_model', 'fitter', 'fitting_ranges', 'fitter_parameters'}))
    warning('HORACE:tobyfit:deprecated_argument', ...
        'The instrument is determined from the sqw object. Redundant option ''%s'' ignored', argi{1})
elseif ~isempty(argi)
    key = struct('res_model', [], 'fitter', [], 'fitting_ranges', [], 'fitter_parameters', []);
    [par, key, ~, ~, ok, mess] = parse_arguments(argi, key);
    if ~ok
        error('HORACE:tobyfit:invalid_argument', '%s', mess)
    elseif ~isempty(par)
        error('HORACE:tobyfit:invalid_argument', ...
            'Unrecognised positional argument(s) passed to tobyfit')
    end
    res_model_arg = key.res_model;
    fitter_arg = key.fitter;
    ranges_arg = key.fitting_ranges;
    fparams_arg = key.fitter_parameters;
end

% Resolve resolution model (explicit override or inferred from sqw instrument)
if isempty(res_model_arg)
    [inst, all_inst] = get_inst_class(sqws{:});
    if isempty(inst)
        if all_inst
            error('HORACE:tobyfit:invalid_argument', 'The instrument type must be the same for all datasets')
        else
            error('HORACE:tobyfit:invalid_argument', 'All sqw objects must now have the instrument type set as an instrument object')
        end
    end
    try
        res_model = tobyfit_get_res_model(inst);
    catch ME
        error('HORACE:tobyfit:invalid_argument', '%s', ME.message)
    end
else
    try
        res_model = tobyfit_get_res_model(res_model_arg);
    catch ME
        error('HORACE:tobyfit:invalid_argument', '%s', ME.message)
    end
end

% Construct mfclass_tobyfit with injected dependencies
ctor_args = {'sqw', 'res_model', res_model};
if ~isempty(fitter_arg)
    ctor_args = [ctor_args, {'fitter', fitter_arg}];
end
if ~isempty(ranges_arg)
    ctor_args = [ctor_args, {'fitting_ranges', ranges_arg}];
end
if ~isempty(fparams_arg)
    ctor_args = [ctor_args, {'fitter_parameters', fparams_arg}];
end
obj = mfclass_tobyfit (sqws{:}, ctor_args{:});

end
