function display_message(mess,opt)
% Display character string or cell array of character strings
%
%   >> display_message (mess)
%   >> display_message (mess, '-squeeze')   % Squeeze out empty lines


% Original author: T.G.Perring
%
% $Revision:: 840 ($Date:: 2020-02-10 16:05:56 +0000 (Mon, 10 Feb 2020) $)


% Get option
if nargin==2
    if is_string(opt) && strncmpi(opt,'-squeeze',numel(opt))
        squeeze=true;
    else
        error('Unrecognised optional argument')
    end
else
    squeeze=false;
end

% Display
if iscellstr(mess) && all(cellfun(@(x)(size(x,1)==1||isempty(x)),mess(:)))
    for i=1:numel(mess)
        if ~squeeze || ~isempty(deblank(mess{i}))
            disp(mess{i})
        end
    end
elseif is_string(mess)
    if ~isempty(deblank(mess))
        disp(mess)
    end
else
    error('Input must be character string or cell array of strings')
end

