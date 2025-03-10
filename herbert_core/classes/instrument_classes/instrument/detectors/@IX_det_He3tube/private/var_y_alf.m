function val = var_y_alf (alf)
% Variance of width of absorption in a 3He tube
%
%   >> val = var_y_alf (alf)
%
% Input:
% ------
%   alf     Maximum path length along neutron direction in a 3He tube as a
%          multiple of the macroscopic absorption cross-section (scalar or array)
%
% Output:
% -------
%   val     Variance of width of aborption as a fraction of the radius
%          squared (same size and shape as alf)
%
%
% History of the algorithm
% ------------------------
%
%  T.G.Perring June 1990:
%
%  Algorithm is based on a combination of Taylor series and
%  assymptotic expansion of the double integral for the
%  efficiency, linearly interpolating betweent the two in
%  region of common accuracy. Checked against numerical
%  integration to yield relative accuracy of 1 part in 10^12
%  or better over the entire domain of the input arguments
%
%  T.G.Perring August 2015:
%
%  Fortran code minimally adapted for Matlab


% Original author: T.G.Perring


c_eff_g = [2.033429926215546,...
    -2.3123407369310212E-02, 7.0671915734894875E-03,...
    -7.5970017538257162E-04, 7.4848652541832373E-05,...
     4.5642679186460588E-05,-2.3097291253000307E-05,...
     1.9697221715275770E-06, 2.4115259271262346E-06,...
    -7.1302220919333692E-07,-2.5124427621592282E-07,...
     1.3246884875139919E-07, 3.4364196805913849E-08,...
    -2.2891359549026546E-08,-6.7281240212491156E-09,...
     3.8292458615085678E-09, 1.6451021034313840E-09,...
    -5.5868962123284405E-10,-4.2052310689211225E-10,...
     4.3217612266666094E-11, 9.9547699528024225E-11,...
     1.2882834243832519E-11,-1.9103066351000564E-11,...
    -7.6805495297094239E-12, 1.8568853399347773E-12];

c_varw_f = [2.408884004758557,...
     0.1441097208627301,    -5.0093583831079742E-02,...
     1.0574012517851051E-02,-4.7245491418700381E-04,...
    -5.6874753986616233E-04, 2.2050994176359695E-04,...
    -3.0071128379836054E-05,-6.5175276460682774E-06,...
     4.2908624511150961E-06,-8.8327783029362728E-07,...
    -3.5778896608773536E-08, 7.6164115048182878E-08,...
    -2.1399959173606931E-08, 1.1599700144859781E-09,...
     1.2029935880786269E-09,-4.6385151497574384E-10,...
     5.7945164222417134E-11, 1.5725836188806852E-11,...
    -9.1953450409576476E-12, 1.7449824918358559E-12,...
     1.2301937246661510E-13,-1.6739387653785798E-13,...
     4.5505543777579760E-14,-4.3223757906218907E-15];
 
c_varw_g = [1.970558139796674,...
     1.9874189524780751E-02,-5.3520719319403742E-03,...
     2.3885486654173116E-04,-4.1428357951582839E-05,...
    -6.3229035418110869E-07, 2.8594609307941443E-06,...
    -8.5378305322625359E-07,-8.2383358224191738E-08,...
     1.1218202137786015E-07,-6.0736651874560010E-09,...
    -1.4453200922748266E-08, 1.7154640064021009E-09,...
     2.1673530992138979E-09,-2.4074988114186624E-10,...
    -3.7678839381882767E-10, 1.1723938486696284E-11,...
     7.0125182882740944E-11, 7.5127332133106960E-12,...
    -1.2478237332302910E-11,-3.8880659802842388E-12,...
     1.7635456983633446E-12, 1.2439449470491581E-12,...
    -9.4195068411906391E-14,-3.4105815394092076E-13];

val=zeros(size(alf));
ilo=(alf<=9);
ihi=(alf>=10);
imix=(alf>9 & alf<10);

if any(ilo(:))
    val(ilo) = 0.25*effchb(0,10,c_varw_f,alf(ilo));
end
     
if any(ihi(:))
    y = 1-18./alf(ihi);
    eff = 1 - effchb(-1,1,c_eff_g,y)./(alf(ihi).^2);
    val(ihi) = (-effchb(-1,1,c_varw_g,y)./(alf(ihi).^2) + 1/3) ./ eff;
end

if any(imix(:))
    varw_f = 0.25*effchb(0,10,c_varw_f,alf(imix));
    y = 1-18./alf(imix);
    eff = 1 - effchb(-1,1,c_eff_g,y)./(alf(imix).^2);
    varw_g = (-effchb(-1,1,c_varw_g,y)./(alf(imix).^2) + 1/3) ./ eff;
    val(imix) = (10-alf(imix)).*varw_f  + (alf(imix)-9).*varw_g;
end

%---------------------------------------------------------------------
function y=effchb(a,b,c,x)
% Essentially CHEBEV of "Numerical Recipes"
d=zeros(size(x)); dd=zeros(size(x)); y=(2*x-a-b)/(b-a); y2=2*y;
for j=numel(c):-1:2
    sv=d;
    d=(y2.*d-dd)+c(j);
    dd=sv;
end
y=(y.*d-dd)+0.5*c(1);

