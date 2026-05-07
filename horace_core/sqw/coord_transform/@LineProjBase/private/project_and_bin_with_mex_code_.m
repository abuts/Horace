function varargout =project_and_bin_with_mex_code_(obj, ...
    axes,in_pix,npix,s,e,varargin)
%PROJECT_AND_BIN_WITH_MEX_CODE_ transform pixels in image coordinate system
% and sort them according to the image bins using mex code.
%
% Inputs:
% obj       -- initialized instance of LineProjBase
% axes      -- correspondent axes linear block
% in_pix    -- PixelDataMemory class to bin. This subalgorithm works in
%              memory only
% npix,s,e  -- n-D arrays accumulators containing binned pixels
%
% Outputs:
%          Depending on binning mode, various results of binning pixels,
%          starting from filled in accumulators and up to properly binned
%          and sorted pixels. See bin_mode class for implemented binning
%          modes.

nout = nargout;
varargout = cell(1,nout);

[ok,mess,force_double,return_selected,test_mex_inputs,argi]=parse_char_options(varargin, ...
    {'-force_double', '-return_selected','-test_mex_inputs'});
if ~ok
    error('HORACE:AxesBlockBase:invalid_argument',mess)
end
if isempty(argi)
    argi = {npix,s,e,in_pix}; % reformat input data into the form, requested by normalize_bin_input
else
    argi = [{npix},{s},{e},{in_pix},argi{:}];
end
% identify binning mode as function of number of input
% arguments
mode = bin_mode.from_narg(nargout-1,test_mex_inputs,return_selected,argi{:});
% convert different input forms into fully expanded common form
[npix,s,err,pix_cand,unique_runid]=...
    axes.normalize_bin_input([4,in_pix.num_pixels],mode,argi{:});
% axes binning bins coordinates and does not use pix_cand
% unldess pix data processing is expected. line_proj always
% use pixel data as input and calculates coordinates
% internally, so we need to provide pix candidates to it
if isempty(pix_cand)
    pix_cand = in_pix;
end

[bin_info,ndata,is_pix]=axes.prepare_mex_code_input(mode, ...
    [],pix_cand,unique_runid,force_double,test_mex_inputs);
% add LineProjBase-specific information to binning code
%coord_transf = obj.transform_pix_to_img(pix_cand);
coord_transf = []; % transformed coordinates are calculated within the
% mex-code on-the fly.
[q_to_img,u_offset]=obj.get_pix_img_transformation(3,pix_cand);
if obj.offset(4)>1.e-11
    u_offset(end+1) = obj.offset(4); % energy scale offset. Currently no
    % projection transform energy scale except shift
end
bin_info.q_to_img = q_to_img;
bin_info.u_offset = u_offset;
bin_info.coord_in = coord_transf;
bin_info.ndata = ndata;
bin_info.is_pix = is_pix;

[varargout{1:nout}]=axes.bin_pixels_with_mex_code(coord_transf,bin_info,...
    npix,s,err,pix_cand,unique_runid,force_double,test_mex_inputs);

end