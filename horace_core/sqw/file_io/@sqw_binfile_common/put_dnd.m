function    obj = put_dnd(obj,varargin)
% Save sqw data into new binary file or fully overwrite an existing file
% and write it as dnd file
%
% store header, which describes file as dnd file
%
% store current sqw-data positions
pos = obj.get_pos_info();
% recalculate positions as if it is a dnd object
if nargin>1
    inobj = varargin{1};
    internal_obj = false;
else
    inobj = obj.sqw_holder_;
    internal_obj  = true;
end
% initialize dnd positions
obj.data_pos_=26;
obj.data_type_ = 'b+';
[obj,dnd_inobj] = obj.init_dnd_info(inobj);
% need to do it for correct application header as it will be sqw otherwise
%obj.sqw_holder_ = dnd_inobj;
if isempty(varargin)
    argi = {dnd_inobj};
else
    argi = [varargin(:);{dnd_inobj}];
end


% put dnd data
obj = put_dnd@binfile_v2_common(obj,argi{:});

%revert sqw positions back to object, keeping internals intact (may cause
% problems with update)
obj = obj.copy_contents(pos,true);
if internal_obj
    obj.sqw_holder_ = inobj;
end

