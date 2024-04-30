%% US g1 fit, GPU
% input: 
    % sIQ: bulk motion removed data
    % PRSSinfo: data acquistion information, including
        % PRSSinfo.FWHM: (X, Y, Z) spatial resolution, Full Width at Half Maximum of point spread function, m
        % PRSSinfo.rFrame: sIQ frame rate, Hz
        % PRSSinfo.f0: Transducer center frequency, Hz
        % PRSSinfo.C: Sound speed in the sample, m/s
        % PRSSinfo.g1nT: g1 calculation sample number
        % PRSSinfo.g1nTau: maximum number of time lag
        % PRSSinfo.SVDrank: SVD rank [low high]
        % PRSSinfo.HPfC:  High pass filtering cutoff frequency, Hz
        % PRSSinfo.NEQ: do noise equalization? 0: no noise equalization; 1: apply noise equalization
        % PRSSinfo.rfnScale: spatial refind scale
 % output:
    % Ms: static component fraction, [nz,nx]
    % Mf: dynamic component fraction, [nz,nx,2], 2: [real,imag]
    % Vx: x-direction velocity component, [nz,nx], mm/s
    % Vz: axial-direction velocity component, [nz,nx], mm/s
    % V=sqrt(Vx.^2+Vz.^2), [nz,nx], mm/s
    % pVz: Vz distribution (sigma-Vz), [nz,nx]
    % R: fitting accuracy, [nz,nx]
    % Vcz: Color Doppler axial velocity, [nz,nx], mm/s
    % GGf: gg fitting results, [nz,nx, nTau]
 % Jianbo Tang, 20190404
function [Ms, Mf, Vx, Vz, V, pVz ,R, GGf]=sIQ2vUS_GPU(sIQ, PRSSinfo)
%% O. constant
lambda0=PRSSinfo.C/PRSSinfo.f0;        % wavlength
k0 = 2*pi/lambda0;   % wave number
PRSSinfo.FWHM=[PRSSinfo.FWHM(1) 1000e-6 PRSSinfo.FWHM(2)]; % just to put the FWHM_y, any number other than 0
Sigma=PRSSinfo.FWHM*0.7/(2*sqrt(2*log(2))); % intensity-based sigma
Sigma2=2*Sigma;
nItpVz0=10;          % for Vz0 determination
dt = 1/PRSSinfo.rFrame;      % frame interval, s
tau = [1:PRSSinfo.g1nTau]*dt; % time lag, s
tn = tau / tau(end);
%% I. sIQ2GG and spatial refine GG
sIQ=gpuArray(sIQ);
[nz0,nx0,nt]=size(sIQ);
PRSSinfo.g1StartT=1;
GG0 = gather(sIQ2GG(sIQ, PRSSinfo));
SNR0=gather(SNR0);
clear sIQ;
% GG = sIQ2GG_GPU(sIQ, PRSSinfo);
[nz0,nx0,nTau]=size(GG0);
if PRSSinfo.rfnScale>1
    for it=1:nTau
        GG(:,:,it)=imresize(GG0(:,:,it),[nz0,nx0]*PRSSinfo.rfnScale,'bilinear'); % spatial interpolation
    end
else
    GG=GG0;
end
GG=gpuArray(GG);
clear GG0 SNR0;
%% III. GG2Vz
[nz,nx,nTau]=size(GG);
PRSSinfo.Dim=[nz,nx,nTau];
GG2=reshape(GG,[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),PRSSinfo.Dim(3)]);
clear GG
[g1Vz0, Tvz]=GG2Vz(GG2, PRSSinfo, 10);
%% IV. vUS initial
Ms0 = min(max(real(FindCOR(GG2(:,floor(end*1/2):end))),0),max(mean(real(GG2(:,floor(end*2/3):end)),2),0));
Me0 =1-abs(GG2(:,1));
MfR0 = max(1-Ms0-Me0,0);
PRSSinfo.MPvz=0;
[Vx0,Vz0,PVz0,MfI0,R0]=iniVx0Vz0Pvz0(GG2, g1Vz0, Ms0, MfR0, PRSSinfo);
%% V. vUS fitting
% V.1. Fitting constraint
Fmin_cstrn(:,:,1)=[Ms0-0.1 Ms0+0.1];   % Ms constrain
Fmin_cstrn(:,:,2)=[max(MfR0-0.15, 0) min(MfR0+0.1,1-Ms0)];   % MfR constrain
Fmin_cstrn(:,:,3)=[max(MfI0-0.15, 0) min(MfI0+0.2,1)];   % MfI constrain
Fmin_cstrn(:,:,4)=[0.5*Vx0 1.3*Vx0]*tau(end)/(Sigma2(1)); % Vx constrain
Fmin_cstrn(:,:,5)=sign(Vz0).*[0.6*abs(Vz0) 1.2*abs(Vz0)];  % Vz constrain, mm/s
Fmin_cstrn(:,:,6)=[0.8*PVz0 min(PVz0*1.3,0.7)]; % PVz constrain
fitC0(:,:,1) = double(Ms0); % initials
fitC0(:,:,2) = double(MfR0); % initials
fitC0(:,:,3) = double(MfI0); % initials
fitC0(:,:,4) = double(Vx0*tau(end)/(Sigma2(1))); % initials
fitC0(:,:,5) = double(Vz0); % initials
fitC0(:,:,6) = double(PVz0); % initials
fitC0=double(gather(fitC0));
Fmin_cstrn=double(gather(Fmin_cstrn));
%% V.2 fit complex (g1)
fitE = @(c) double(sum( abs(c(:,1,1) + c(:,1,2).*exp( -(c(:,1,4).*tn).^2-(c(:,1,5).*tau).^2/(Sigma2(3))^2).*exp(-(k0*tau.*c(:,1,5).*c(:,1,6)).^2).*cos(2*k0*c(:,1,5).*tau)+...
    1i*c(:,1,3).*exp( -(c(:,1,4).*tn).^2-(c(:,1,5).*tau).^2/(Sigma2(3))^2).*exp(-(k0*tau.*c(:,1,5).*c(:,1,6)).^2).*sin(2*k0*c(:,1,5).*tau)- (GG2) ).^2 ,2));
[fitC, fval] = fmincon(fitE, fitC0, [],[],[],[], ...
    [Fmin_cstrn(:,1,1) Fmin_cstrn(:,1,2) Fmin_cstrn(:,1,3) Fmin_cstrn(:,1,4) Fmin_cstrn(:,1,5) Fmin_cstrn(:,1,6)], ...
    [Fmin_cstrn(:,2,1) Fmin_cstrn(:,2,2) Fmin_cstrn(:,2,3) Fmin_cstrn(:,2,4) Fmin_cstrn(:,2,5) Fmin_cstrn(:,2,6)], ...
    [], optimset('Display','off','TolFun',1e-6,'TolX',1e-6));%

Ms=reshape(fitC(:,1,1),[nz,nx]); 
Mf(:,:,1)=reshape(fitC(:,1,2),[nz,nx]);  % MfR
Mf(:,:,2)=reshape(fitC(:,1,3),[nz,nx]);  % MfI
Vx=reshape(fitC(:,1,4),[nz,nx])/(tau(end)/(Sigma2(1))); % m/s
Vz=reshape(fitC(:,1,5),[nz,nx]);  % m/s
V=sqrt(Vz.^2+Vx.^2);
pVz=reshape(fitC(:,1,6),[nz,nx]); % 
GGf0=fitC(:,:,1) + fitC(:,:,2).*exp( -(fitC(:,:,4).*tn).^2-(fitC(:,:,5).*tau).^2/(Sigma2(3))^2).*exp(-(k0*tau.*fitC(:,:,5).*fitC(:,:,6)).^2).*cos(2*k0*fitC(:,:,5).*tau)+...
    1i*fitC(:,:,3).*exp( -(fitC(:,:,4).*tn).^2-(fitC(:,:,5).*tau).^2/(Sigma2(3))^2).*exp(-(k0*tau.*fitC(:,:,5).*fitC(:,:,6)).^2).*sin(2*k0*fitC(:,:,5).*tau);
R=gather((reshape((1-sum(abs(GG2-GGf0).^2,2)./sum(abs((GG2)-mean(GG2,2)).^2,2)),[nz,nx])));  
GGf=gather(reshape(GGf0,[nz,nx,nTau]));
V=gather(V*1e3); Vz=gather(Vz*1e3); Vx=gather(Vx*1e3) ;% mm/s
Mf=gather(Mf); Ms=gather(Ms); pVz=gather(pVz);            
