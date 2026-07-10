function dump_oracle()
% Dump intermediate quantities from the MATLAB AstroSeis kernels so the
% Python port can be unit-tested piece by piece (tests/test_oracle.py).
% Small enough to run on a login node (seconds of compute).
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(genpath(fullfile(root,'lib_BEM')));

% keep parfor serial (no parallel pool on login nodes)
try
    ps = parallel.Settings; ps.Pool.AutoCreate = false;
catch
end

load(fullfile(root,'my_mesh.mat'),'face');

vpr=6000; vsr=3000; rho=3000; Q=570;
mu0 = rho*vsr*vsr;
lamda0 = rho*vpr*vpr - 2*mu0;
Qp = 2.5*Q;
vp = vpr/(1+complex(0,1)*0.5/Qp);
vs = vsr/(1+complex(0,1)*0.5/Q);
w = 2*pi*0.02 + complex(0,1)*0.08;

%% 1. raw traction kernel: field pts = centroids of faces 101:200,
%%    normals of those faces, source at face(7) incenter
idx = 101:200; np_ = numel(idx);
x1=zeros(1,np_); x2=x1; x3=x1; n1=x1; n2=x1; n3=x1;
for k=1:np_
    i = idx(k);
    x1(k)=face(i).ic(1); x2(k)=face(i).ic(2); x3(k)=face(i).ic(3);
    n1(k)=face(i).nvec(1); n2(k)=face(i).nvec(2); n3(k)=face(i).nvec(3);
end
ps7 = face(7).ic;
[T11,T12,T13,T21,T22,T23,T31,T32,T33] = green_traction_tensor(vp,vs,rho,w,...
    x1,x2,x3,ps7(1),ps7(2),ps7(3),n1,n2,n3);

%% 2. sub-grid quadrature for face 3
[sgx,sgy,sgz,~,~,~,sgw] = subgrid_gen(face(3),10);

%% 3. self integral for face 5 (grid replicated from cal_traction_tri_vec)
nxi=300; ridfac=1;
xg = linspace(0,1,nxi); dxg = xg(2)-xg(1);
etag = linspace(0,1,nxi);
[xxg,nng] = meshgrid(xg,etag);
xg1 = xxg(:); eg1 = nng(:);
ind = find(xg1+eg1 <= 1);
refs = [xg1(ind).'; eg1(ind).'];
wis = dxg*dxg;
ps5 = face(5).ic;
ST5 = int_self_trac_tri_vec(vp,vs,rho,w,ps5(1),ps5(2),ps5(3),face(5),wis,refs,ridfac);

%% 4. small full system on a 60-face subset + source term + solve
fsub = face(1:60);
TRACs = cal_traction_tri_vec(fsub,60,w,lamda0,mu0,rho,Q);

for i=1:length(face), Ra(i)=norm(face(i).ic); end
R = mean(Ra);
h = 2e3; rs = R-h; thetas = pi/2; phis = pi;
xs = rs*sin(thetas)*cos(phis);
ys = rs*sin(thetas)*sin(phis);
zs = rs*cos(thetas);
u0s = u0e_func(fsub,w,rho,mu0,lamda0,xs,ys,zs,Q,[1 1 1]);
xsol = TRACs\u0s;

save(fullfile(here,'oracle.mat'),'T11','T12','T13','T21','T22','T23',...
    'T31','T32','T33','sgx','sgy','sgz','sgw','ST5','TRACs','u0s','xsol',...
    'w','vp','vs','R','xs','ys','zs','-v7');
fprintf('oracle.mat written\n');
end
