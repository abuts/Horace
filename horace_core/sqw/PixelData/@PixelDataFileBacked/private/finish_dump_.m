function obj = finish_dump_(obj,page_op)
% complete pixel write operation, close writing to the target file and
% open pixel dataset for read operations.
%
%
if nargin>1 % new interface
    wh = page_op.write_handle;
    init_info = wh.release_pixinit_info();
    if wh.is_tmp_file
        obj.tmp_file_holder_ = TmpFileHandler(wh.write_file_name,true);
    end
    obj = obj.init(init_info);
    return
end

% Re #1302 TODO: DELETE Old interface. Letf to maintain the tests which
% are not yet transformed as part of #1302.
if ~obj.has_open_file_handle
    error('HORACE:PixelDataFileBacked:runtime_error', ...
        'Cannot finish dump writing, object does not have open filehandle')
end

obj.num_pixels_ = obj.write_handle_;

if isa(obj.write_handle_, 'sqw_file_interface')
    obj.full_filename = obj.write_handle_.full_filename;
    obj.write_handle_ = obj.write_handle_.put_pix_metadata(obj);
    % Force pixel update
    obj.write_handle_ = obj.write_handle_.put_num_pixels(obj.num_pixels);

    obj = init_from_file_accessor_(obj,obj.write_handle_, false, true);
    obj.write_handle_.delete();
    obj.write_handle_ = [];

else
    if ~isempty(obj.write_handle_)
        fclose(obj.write_handle_);
    end
    if obj.num_pixels_ == 0
        obj = PixelDataMemory();
        return;
    end

    obj.write_handle_ = [];
    obj.f_accessor_ = [];
    obj.offset_ = 0;
    obj.full_filename = obj.tmp_file_holder_.file_name;
    obj.f_accessor_ = memmapfile(obj.full_filename, ...
        'format', obj.get_memmap_format(), ...
        'Repeat', 1, ...
        'Writable', true, ...
        'offset', obj.offset_);
end
