function dump_oracle2()
% Oracle dump #2: moment-tensor source, liquid-core BEM stack, and mesh
% generation primitives. Small subsets only (seconds of compute).
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(genpath(fullfile(root,'lib_BEM')));
try
    ps = parallel.Settings; ps.Pool.AutoCreate = false;
catch
end

load(fullfile(root,'examples','shifted_liquid_core','layers_shift.mat'),'layer');
f2full = layer(1).face;   % core interface
f1full = layer(2).face;   % free surface
f2 = f2full(1:40);
f1 = f1full(1:60);

vp1=6000; vs1=3000; rho1=3000; Q=200;
vp2=8000; vs2=0; rho2=4000;
mu1 = rho1*vs1*vs1;  lamda1 = rho1*vp1*vp1 - 2*mu1;
mu2 = rho1*vs2*vs2;  lamda2 = rho2*vp2*vp2 - 2*mu2;
w  = 2*pi*0.04 + complex(0,1)*0.08;
w0 = 2*pi*0.3;
nint = 10; nxi = 300;

% complex speeds, legacy factors
Qp = 2.5*Q;
vpc1 = vp1/(1+complex(0,1)*0.5/Qp);
vsc1 = vs1/(1+complex(0,1)*0.5/Q);
vpc2 = vp2/(1+complex(0,1)*0.5/Qp);

%% 1. kernels at field points = f1(1:50) incenters, source = f2(3) incenter
npk = 50;
x1=zeros(1,npk); x2=x1; x3=x1; n1=x1; n2=x1; n3=x1;
for k=1:npk
    x1(k)=f1(k).ic(1); x2(k)=f1(k).ic(2); x3(k)=f1(k).ic(3);
    n1(k)=f1(k).nvec(1); n2(k)=f1(k).nvec(2); n3(k)=f1(k).nvec(3);
end
psrc = f2(3).ic;
[B11,B12,B13,B21,B22,B23,B31,B32,B33] = Bmat_func(vpc1,vsc1,rho1,w,...
    x1,x2,x3,psrc(1),psrc(2),psrc(3));
gp  = greens_func_p(vpc2,w,x1,x2,x3,psrc(1),psrc(2),psrc(3));
gdp = greens_func_deri_p(vpc2,w,x1,x2,x3,psrc(1),psrc(2),psrc(3),n1,n2,n3);
GSV = G_singular_value(vpc1,vsc1,rho1,w,f1(5).r,...
    f1(5).nvec(1),f1(5).nvec(2),f1(5).nvec(3));
gd1 = zeros(3,3,20); gd2 = gd1; gd3 = gd1;
for k=1:20
    [a,b,c] = greens_deri_src(vpc1,vsc1,rho1,w,...
        x1(k),x2(k),x3(k),psrc(1),psrc(2),psrc(3));
    gd1(:,:,k)=a; gd2(:,:,k)=b; gd3(:,:,k)=c;
end

%% 2. liquid-core interaction matrices on subsets
f2w = f2;
for i=1:length(f2), f2w(i).nvec = -f2(i).nvec; end
T11s = cal_T_st(f1, f1, w,lamda1,mu1,rho1,Q,nint,nxi);
T22s = cal_T_st(f2w,f2w,w,lamda1,mu1,rho1,Q,nint,nxi);
T12s = cal_T_st(f1, f2w,w,lamda1,mu1,rho1,Q,nint,nxi);
T21s = cal_T_st(f2w,f1, w,lamda1,mu1,rho1,Q,nint,nxi);
G12s = cal_G_st(f1, f2w,w,lamda1,mu1,rho1,Q,nint,nxi);
G22s = cal_G_st(f2w,f2w,w,lamda1,mu1,rho1,Q,nint,nxi);
A22s = cal_A_st(f2, f2, w,lamda2,mu2,rho2,Q,nint,nxi);
B22s = cal_B_st(f2, f2, w,lamda2,mu2,rho2,Q,nint,nxi);

%% 3. moment source + pressure source + full small coupled system
for i=1:length(f1full), Ra(i)=norm(f1full(i).ic); end
R = mean(Ra); h = 4e3; rs = R-h; thetas = pi/2; phis = pi;
xs = rs*sin(thetas)*cos(phis);
ys = rs*sin(thetas)*sin(phis);
zs = rs*cos(thetas);
M = [1 2 3; 2 4 5; 3 5 6];
u01m = u0eM_func(f1,w,rho1,mu1,lamda1,xs,ys,zs,Q,M);
u02m = u0eM_func(f2,w,rho1,mu1,lamda1,xs,ys,zs,Q,M);
u0p  = u0p_func_exp(f2,w,vp2,xs,ys,zs,Q);
P0 = zeros(length(f2),1);
Smats = Smat_func(f2);
[Alc,blc] = liq_core(f1,f2,w,lamda1,mu1,rho1,lamda2,mu2,rho2,Q,nint,...
    nxi,P0,u01m,u02m,Smats,w0);
xlc = Alc\blc;

%% 4. spherical harmonics + topography
tt = linspace(0.05,pi-0.05,25);
pp = linspace(-3,3,25);
Y45  = ylm1(45,17,tt,pp);
Y5   = ylm1(5,3,tt,pp);
Y7m  = ylm1(7,-4,tt,pp);
topo25 = get_phobos_topo(tt,pp);
load(fullfile(root,'lib_BEM','mesh','marstopo.mat'));
llat2 = 90 - llat;
topos_mars = smooth_topo_rand(llon, llat2, aa, 8);

%% 5. MATLAB rng(4) rand stream (embedded as package data)
rng('default'); rng(4);
coefrand500 = rand(500,1);
fid = fopen(fullfile(root,'pyastroseis','data','coefrand_rng4.txt'),'w');
fprintf(fid,'%.17g\n',coefrand500);
fclose(fid);

%% 6. deterministic subdivision on a fixed icosahedron
gr = (1+sqrt(5))/2;
VI = [0 1 gr; 0 -1 gr; 0 1 -gr; 0 -1 -gr;
      1 gr 0; -1 gr 0; 1 -gr 0; -1 -gr 0;
      gr 0 1; gr 0 -1; -gr 0 1; -gr 0 -1];
VI = VI./vecnorm(VI,2,2);
FI = fliplr(convhulln(VI));
fv = SubdivideSphericalMesh(struct('faces',FI,'vertices',VI),2);
Vsub = fv.vertices; Fsub = fv.faces;

save(fullfile(here,'oracle2.mat'),'B11','B12','B13','B21','B22','B23',...
    'B31','B32','B33','gp','gdp','GSV','gd1','gd2','gd3',...
    'T11s','T22s','T12s','T21s','G12s','G22s','A22s','B22s',...
    'u01m','u02m','u0p','Alc','blc','xlc','xs','ys','zs','M','w','w0',...
    'Y45','Y5','Y7m','topo25','topos_mars','VI','FI','Vsub','Fsub','-v7');
fprintf('oracle2.mat written\n');
end
