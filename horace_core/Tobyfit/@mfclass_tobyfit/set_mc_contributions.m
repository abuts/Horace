function obj = set_mc_contributions (obj, varargin)
% Set the ocomponents that contribute to the resolution function
%
% Set default (all contributions):
%   >> obj = obj.set_mc_contributions
%
% Set all contributions or no contributions:
%   >> obj = obj.set_mc_contributions ('all')    % all contributions
%   >> obj = obj.set_mc_contributions ('none')   % no contributions
%
% All components included except...
%   >> obj = obj.set_mc_contributions ('nomoderator')
%   >> obj = obj.set_mc_contributions ('nomoderator','nochopper')
%
% Only include...
%   >> obj = obj.set_mc_contributions ('chopper','sample')
%
% Set according to a previously recovered contributions structure
%   >> mc_contr = obj.mc_contributions;
%       :
%   >> obj = obj.set_mc_contributions (mc_contr)
%
% Modify a previously recovered contributions structure
%   >> mc_contr = obj.mc_contributions;
%       :
%   >> obj = obj.set_mc_contributions (mc_contr,'nochop','sample')


% Get the possible contributions
if ~isempty(obj.res_model_)
    mc_contr = obj.res_model_.mc_contribution_names();
elseif ~isempty(obj.wrapfun.func_init)
    [~,~,mc_contr] = obj.wrapfun.func_init();
else
    mc_contr = {};
end

% Now parse input
[mc_contributions,ok,mess] = mc_contributions_parse (mc_contr,varargin{:});
if ok
    obj.mc_contributions_ = mc_contributions;
else
    error('HORACE:mfclass_tobyfit:invalid_argument', '%s', mess)
end
