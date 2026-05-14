function [in_code_struct,ndata,is_pix,mode_to_bin] = prepare_mex_code_input_(obj, ...
    mode_to_bin,coord,pix_cand,unique_runid,force_double,test_mex_inputs)
% Prepare structure-input for binning procedure which uses C++ mex code
%
% Inputs:
% obj         -- instance of AxesBlockBase binning code
% mode_to_bin -- particular binning mode as in bin_mode enum.
% coord       -- the 3D or 4D array of pixels coordinates transformed
%                into AxesBlockBase coordinate system
% pix_cand    -- input data to bin. Comes in different forms
%                and transformed into specific form acceptable
%                by mex code
% force_double -- if true, the routine changes type of pixels
%                 it gets on input, into double. if not, output
%                 pixels will keep their initial type.
% return_selected
%              -- if true sets pix_ok to return the indices of selected
%                 pixels for use with DnD cuts where fewer args are
%                 requested
% SPECIAL:
% test_mex_inputs
%              -- if ture, routine works in testing mode and all input
%                 parameters are reflected to output parameters.
%                 This mode used in unit testing to verify correct
%                 operations of mex code.
% Returns:
% in_code_struct
%             -- structure to be parsed by mex code at input.
%                see below the description of its fields.
% ndata       -- number which defines type of mex code output
% is_pix      -- if true, input privided in pixels form, false --
%                array or cellarray of input data
%
% NOTE: made public to be able to reuse and overload it in
% aProjectionBase
%
%
if isstruct(mode_to_bin)
    %Slave mode  All parameters were already calculated and stored within
    %the mode structure
    ndata = mode_to_bin.ndata;
    is_pix= true;
    in_code_struct = rmfield(mode_to_bin,{'ndata','is_pix'});
    mode_to_bin = bin_mode(mode_to_bin.binning_mode);
    return
end

[num_threads,pix_omp_limit,dynamic_omp_stride] = config_store.instance().get_value( ...
    'parallel_config','threads','min_npix_for_omp_cut','dynamic_omp_npixels_stride');
[ignore_nan,ignore_inf]  = config_store.instance().get_value( ...
    'hor_config','ignore_nan','ignore_inf');
nbins_all_dims_in = uint32(obj.nbins_all_dims(:)');
if size(coord,1) == 3  % 3D array binning
    pax_in = obj.pax;
    data_range     = obj.img_range(:,1:3);
    nbins_all_dims_in = nbins_all_dims_in(1:3);
    pax_in = pax_in(pax_in~=4);
    ndims = numel(pax_in);
else
    data_range = obj.img_range;
    ndims = obj.dimensions;
end

in_code_struct = struct( ...
    'coord_in',    coord,...                % input coordinates to bin. May be empty in modes when they are processed from transformed pixel data
    'binning_mode',double(mode_to_bin), ... % binning mode, what binning values to calculate and return
    'num_threads', num_threads,  ...        % how many threads to use in parallel computation
    'dynamic_omp_stride',dynamic_omp_stride,... % control static/dynamic scheduling and dynamic scheduler stride
    'data_range',  data_range,...           % binning ranges
    'dimensions',   ndims, ...              % number of image dimensions (sum(nbins_all_dims > 1)))
    'nbins_all_dims',nbins_all_dims_in, ... % dimensions of binning lattice
    'unique_runid', uint32(unique_runid), ... % unique run indices of pixels contributing into cut
    'force_double', force_double, ...       % make result double precision regardless of input data
    'ignore_nan'  , ignore_nan,...
    'ignore_inf'  , ignore_inf,...
    'q_to_img',[],...                       % transformation matrix used in projection binning
    'u_offset',[],...                       % offset used in projection binning
    'test_input_parsing',test_mex_inputs ...% Run mex code in test mode validating the way input have been parsed by mex code and doing no caclculations.
    );

is_pix = isa(pix_cand,'PixelDataBase');
if is_pix
    in_code_struct.pix_candidates   = pix_cand.get_raw_data;
    in_code_struct.check_pix_selection = true; % check if pixels have already been processed by previous symmetry operations
    ndata = 2;
    if pix_cand.is_corrected
        in_code_struct.alignment_matr = pix_cand.alignment_matr;
    else
        in_code_struct.alignment_matr  = [];
    end
    disable_omp = pix_cand.num_pixels < pix_omp_limit;
else
    if iscell(pix_cand)
        % this may be improved/simplified in a future by enabling mex code
        % to process any type of data
        for i=1:numel(pix_cand)
            if ~isa(pix_cand{i},'double')
                pix_cand{i} = double(pix_cand{i});
            end
        end
        disable_omp = true; %  TODO: OMP is not enabled for binning cells of pixels yet.
    else
        disable_omp = size(coord,2) < pix_omp_limit;
    end
    in_code_struct.alignment_matr   = [];
    in_code_struct.pix_candidates   = pix_cand;
    in_code_struct.check_pix_selection  = false; % use all pixels, do not analyze selection
    % cell with data array
    ndata = numel(pix_cand);
end
if disable_omp
    in_code_struct.num_threads = 1;
end

end
