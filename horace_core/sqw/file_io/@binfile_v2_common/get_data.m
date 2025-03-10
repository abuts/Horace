function [data_str,obj] = get_data (obj,varargin)
% Read the data block from an sqw or dnd file and return the result as the
% data structure with fields, described below.
%
% The file pointer is left at the end of the data block.
%
%   >> data = obj.get_data()
%   >> data = obj.get_data(opt)
%
% Input:
% ------
%   opt         [optional] Determines which fields to read
%                  '-hverbatim'    Same as '-h' except that the file name as stored in the main_header and
%                                  data sections are returned as stored, not constructed from the
%                                  value of fopen(fid). This is needed in some applications where
%                                  data is written back to the file with a few altered fields.
%                  '-head'
%                  '-verbatim'
%
%
% Output:
% -------

%   data        Output data structure actually read from the file. Will be one of:
%                   type 'h'    fields: fields: uoffset,...,dax[,img_db_range]
%                   type 'b'    fields: filename,...,dax,s,e
%                   type 'b+'   fields: filename,...,dax,s,e,npix
%                   type 'a'    fields:  filename,...,dax,s,e,npix,img_db_range,pix  (never produced by this reader)
%                   type 'a-'   fields: filename,...,dax,s,e,npix,img_db_range
%               The final field img_db_range is present for type 'h' if the header information was read from an sqw-type file.
%
%
%
% Fields read from the file are:
% ------------------------------
%   data.filename   Name of sqw file that is being read, excluding path
%   data.filepath   Path to sqw file that is being read, including terminating file separator
%          [Note that the filename and filepath that are written to file are ignored; we fill with the
%           values corresponding to the file that is being read.]
%
%   data.title      Title of sqw data structure
%   data.alatt      Lattice parameters for data field (Ang^-1)
%   data.angdeg     Lattice angles for data field (degrees)
%   data.uoffset    Offset of origin of projection axes in r.l.u. and energy ie. [h; k; l; en] [column vector]
%   data.u_to_rlu   Matrix (4x4) of projection axes in hkle representation
%                      u(:,1) first vector - u(1:3,1) r.l.u., u(4,1) energy etc.
%   data.ulen       Length of projection axes vectors in Ang^-1 or meV [row vector]
%   data.ulabel     Labels of the projection axes [1x4 cell array of character strings]
%   data.iax        Index of integration axes into the projection axes  [row vector]
%                  Always in increasing numerical order
%                       e.g. if data is 2D, data.iax=[1,3] means summation has been performed along u1 and u3 axes
%   data.iint       Integration range along each of the integration axes. [iint(2,length(iax))]
%                       e.g. in 2D case above, is the matrix vector [u1_lo, u3_lo; u1_hi, u3_hi]
%   data.pax        Index of plot axes into the projection axes  [row vector]
%                  Always in increasing numerical order
%                       e.g. if data is 3D, data.pax=[1,2,4] means u1, u2, u4 axes are x,y,z in any plotting
%                                       2D, data.pax=[2,4]     "   u2, u4,    axes are x,y   in any plotting
%   data.p          Cell array containing bin boundaries along the plot axes [column vectors]
%                       i.e. row cell array{data.p{1}, data.p{2} ...} (for as many plot axes as given by length of data.pax)
%   data.dax        Index into data.pax of the axes for display purposes. For example we may have
%                  data.pax=[1,3,4] and data.dax=[3,1,2] This means that the first plot axis is data.pax(3)=4,
%                  the second is data.pax(1)=1, the third is data.pax(2)=3. The reason for data.dax is to allow
%                  the display axes to be permuted but without the contents of the fields p, s,..pix needing to
%                  be reordered [row vector]
%   data.s          Cumulative signal.  [size(data.s)=(length(data.p1)-1, length(data.p2)-1, ...)]
%   data.e          Cumulative variance [size(data.e)=(length(data.p1)-1, length(data.p2)-1, ...)]
%   data.npix       No. contributing pixels to each bin of the plot axes.
%                  [size(data.pix)=(length(data.p1)-1, length(data.p2)-1, ...)]
%
%   data.img_db_range  The range of the image along each axis [img_db_range(2,4)]
%
% NOTES:
% ======
% Supported file Formats
% ----------------------
% The current sqw file format comes in two variants:
%   - Horace version 1 and version 2: file format '-v2'
%     (Autumn 2008 onwards). Does not contain instrument and sample fields in the header block.
%     This format is the one still written if these fields all have the 'empty' value in the sqw object.
%
%
% Original author: T.G.Perring
%

% Initialise output arguments
[ok,mess,header_only,verbatim,hverbatim,noclass,noupgrade,~]=...
    parse_char_options(varargin, ...
    {'-head','-verbatim','-hverbatim','-noclass','-noupgrade'});
if ~ok
    error('HORACE:binfile_v2_common:invalid_argument',...
        'get_data::Error: %s',mess);
end
header_only = header_only||hverbatim;
verbatim    = verbatim||hverbatim;


% --------------------------------------------------------------------------
% Read data
% --------------------------------------------------------------------------
% This first set of fields are required for all output options
% ------------------------------------------------------------
if ischar(obj.num_dim)
    error('HORACE:binfile_v2_common:runtime_error',...
        'get_data: method called on un-initialized loader')
end

try
    do_fseek(obj.file_id_,obj.data_pos_,'bof');
catch ME
    exc = MException('COMBINE_SQW_PIX_JOB:io_error',...
        'Can not move to the start of the main data block');
    throw(exc.addCause(ME))
end

sz = obj.s_pos_ - obj.data_pos_+1;
bytes = fread(obj.file_id_,sz,'*uint8');
check_error_report_fail_(obj,...
    'get_data: Can not read the main data block');


data_form = obj.get_dnd_form('-header');
data_str = obj.sqw_serializer_.deserialize_bytes(bytes,data_form,1);
clear bytes;
% Compatibility with new(01/12/2020) data_sqw_dnd constructor
if isempty(data_str.iax)
    data_str.iax = zeros(1,0);
end
if isempty(data_str.iint)
    data_str.iint = zeros(2,0);
end
if isempty(data_str.p)
    data_str.p = cell(1,0);
end
% End compatibility block

if ~verbatim
    data_str.filepath = obj.filepath;
    data_str.filename = obj.filename;
end
%
if ischar(obj.num_dim_)
    obj.num_dim_ = numel(data_str.p);
end
if ischar(obj.dnd_dimensions_)
    obj.dnd_dimensions_ = cellfun(@(x)(numel(x)-1),data_str.p,'UniformOutput',true);
end

% convert to double if necessary
if obj.convert_to_double
    data_str = binfile_v2_common.do_convert_to_double(data_str);
end
%data_str = line_axes.convert_old_struct_into_nbins(data_str);
%
if ~header_only
    data_str = obj.get_se_npix(data_str);
end

if header_only || noclass
    data_str.img_range = line_axes.calc_img_db_range(data_str );
    data_str.dimensions = numel(data_str.p);
    return;
end
%

data_str.warn_on_legacy_data = false;
proj = line_proj.get_from_old_data(data_str);
ax   = line_axes.get_from_old_data(data_str);
cname = class(obj);
if contains(cname,'dnd') % if loading dnd object only, remove inf if present in img_range
    ax.img_range = replace_inf_range(ax.img_range);
end

% really old file format does not contain lattice
% in real life we will probably never face this file format
if ~proj.alatt_defined
    proj.alatt = 2*pi;
end
if ~proj.angdeg_defined
    proj.angdeg = 90;
end


data_str = DnDBase.dnd(ax,proj,data_str.s,data_str.e,data_str.npix);
obj.sqw_holder_ = data_str;

end

