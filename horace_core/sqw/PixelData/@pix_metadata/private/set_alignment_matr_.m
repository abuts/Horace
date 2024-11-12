function  obj = set_alignment_matr_(obj,val)
%SET_ALIGNMENT_MATR_ helper property which checks and sets alignment matrix
%
if isempty(val)
    obj.alignment_matr_ = eye(3);
    if obj.is_misaligned_
        % nullify pixel range as it is invalid any more
        obj.data_range_(:,1:3) = PixelDataBase.EMPTY_RANGE_();
    end
    obj.is_misaligned_  = false;
    return;
end
if ~isnumeric(val)
    error('HORACE:pix_metadata:invalid_argument', ...
        'Alignment matrix must be 3x3 numeric matrix. Attempt to set value of class: %s', ...
        class(val));
end
if any(size(val) ~= [3,3])
    error('HORACE:pix_metadata:invalid_argument', ...
        'Alignment matrix must be 3x3 matrix. Attempt to set the matrix of size: %s', ...
        mat2str(val))
end
%
difr = val - eye(3);
if max(abs(difr(:))) > 1.e-9
    obj.alignment_matr_ = val;
    obj.is_misaligned_  = true;
else
    obj.alignment_matr_ = eye(3);
    obj.is_misaligned_  = false;
end