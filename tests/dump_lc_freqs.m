function dump_lc_freqs()
% Run the CURRENT MATLAB liquid-core code at three frequencies with the
% inp_sft_lc parameters and save the solution columns, to arbitrate
% between the Python port and the shipped 2020-era reference output.
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(genpath(fullfile(root,'lib_BEM')));

load(fullfile(root,'examples','shifted_liquid_core','layers_shift.mat'),'layer');
face1 = layer(2).face;   % free surface
face2 = layer(1).face;   % core interface

vp1=6000; vs1=3000; rho1=3000; Q=200;
vp2=8000; vs2=0; rho2=4000;
nt=500; dt=0.1; f0=0.3;
T=nt*dt; wi=4/T; df=1/(nt*dt);
mu1 =rho1*vs1*vs1;
lamda1 = rho1*vp1*vp1 - 2*mu1;
mu2 =rho1*vs2*vs2;                    % driver quirk (rho1), replicated
lamda2 = rho2*vp2*vp2 - 2*mu2;
nint=10; nxi=300;

M = 10^0*[1 0 0; 0 1 0; 0 0 1];      % inp_sft_lc moment line: 1 0 0 1 0 1
h = 4e3; thetas = (90-0)*pi/180; phis = 180*pi/180;
for i=1:length(face1), Ra(i)=mean(norm(face1(i).ic)); end
R = mean(Ra);
rs = R-h;
xs=rs*sin(thetas)*cos(phis);
ys=rs*sin(thetas)*sin(phis);
zs=rs*cos(thetas);

Smat = Smat_func(face2);
iws = [2 16 30];
xcols = cell(1,numel(iws));
for k=1:numel(iws)
    iw = iws(k);
    tic
    w = 2*pi*(iw-1)*df + wi*complex(0,1);
    u02 = u0eM_func(face2,w,rho1,mu1,lamda1,xs,ys,zs,Q,M);
    u01 = u0eM_func(face1,w,rho1,mu1,lamda1,xs,ys,zs,Q,M);
    P0 = zeros(length(face2),1);
    [A,b] = liq_core(face1,face2,w,lamda1,mu1,rho1,lamda2,mu2,rho2,Q,...
        nint,nxi,P0,u01,u02,Smat,2*pi*f0);
    x = A\b;
    xcols{k} = x;
    fprintf('iw=%d done\n', iw); toc
end
xcol_a = xcols{1}; xcol_b = xcols{2}; xcol_c = xcols{3};
save(fullfile(here,'lc_freqs.mat'),'xcol_a','xcol_b','xcol_c','iws','-v7.3');
fprintf('lc_freqs.mat written\n');
end
