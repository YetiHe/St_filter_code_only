%% US g1 fit, SV model, GPU
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
    % Mf: dynamic component fraction, [nz,nx,2], 2: [real,imag]
    % Vx: x-direction velocity component, [nz,nx], mm/s
    % Vz: axial-direction velocity component, [nz,nx], mm/s
    % V=sqrt(Vx.^2+Vz.^2), [nz,nx], mm/s
    % pVz: Vz distribution (sigma-Vz), [nz,nx]
    % R: fitting accuracy, [nz,nx]
    % Ms: static component fraction, [nz,nx]
    % CR: freqCR.*ggCR
    % GGf: gg fitting results, [nz,nx, nTau]
% subfunction:
    % GG = sIQ2GG(sIQ, PRSSinfo)
    % RotCtr = FindCOR(GG)
    % [Vz, Tvz]=GG2Vz(GG, PRSSinfo, nItp)
    % [Vz,Vx,pVz,Ms,Mf,R, GGf]=GG2vUS(GG, Vz0, Ms0, MfR0, PRSSinfo)
 % Jianbo Tang, 20190821
function [Mf, Vx, Vz, V, pVz ,R, Ms, CR, GGf]=sIQ2vUS_SV_GPU(sIQ, PRSSinfo)
%% O. constant
lambda0=PRSSinfo.C/PRSSinfo.f0;        % wavlength
k0 = 2*pi/lambda0;                     % wave number
PRSSinfo.FWHM=[PRSSinfo.FWHM(1) 900e-6 PRSSinfo.FWHM(2)];  % just to put the FWHM_y, any number other than 0
Sigma=PRSSinfo.FWHM*0.7/(2*sqrt(2*log(2)));                 % intensity-based sigma
Sigma2=2*Sigma;
nItpVz0=10;                            % for Vz0 determination
dt = 1/PRSSinfo.rFrame;                % frame interval, s
tau = [1:PRSSinfo.g1nTau]*dt;          % time lag, s
tn = tau / tau(end);
%% I. signal-to-noise ratio of sIQ
sIQ=gpuArray(sIQ);
[nz0,nx0,nt]=size(sIQ);
fCoor=linspace(-PRSSinfo.rFrame/2,PRSSinfo.rFrame/2,nt)';
fCoorSig=zeros(size(fCoor));
% fCoorSig(abs(fCoor)<800)=1;          % signal frequency range
fCoorSig(abs(fCoor)<1800)=1;           % signal frequency range
fCoorSig=circshift(fCoorSig,nt/2);
fIQ=(fft(sIQ,nt,3));                   % no fft shift
SNR0=squeeze(sum(abs(fIQ.*repmat(permute(fCoorSig,[3 2 1]),[nz0 nx0 1])),3))./squeeze(sum(abs(fIQ),3)); % SNR of oringla data
clear fIQ; 
%% II. sIQ2GG and spatial refine GG and SNR
PRSSinfo.g1StartT=1;
PRSSinfo.MpVz=0;
GG =sIQ2GG(sIQ, PRSSinfo);
clear sIQ;
% GG = sIQ2GG_GPU(sIQ, PRSSinfo);
[nz0,nx0,nTau]=size(GG);
SNR=(SNR0>mean(SNR0(:))+1.5*std(SNR0(:)));
if nz0*nx0==1
    SNR=1;
end
clear GG0 SNR0;
CR0=(abs(GG(:,:,1))>0.2).*(SNR);
%% III. GG2Vz
[nz,nx,nTau]=size(GG);
PRSSinfo.Dim=[nz,nx,nTau];
GG2=reshape(GG,[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),PRSSinfo.Dim(3)]);
clear GG
%% IV. vUS initial
Ms0 = min(max(real(FindCOR(GG2(:,floor(end*1/2):end))),0),max(mean(real(GG2(:,floor(end*2/3):end)),2),0));
Me0 =1-abs(GG2(:,1));  MfR0 = max(1-Ms0-Me0,0);
[g1Vz0, Tvz]=GG2Vz(GG2, PRSSinfo, 10);
% [g1Vz0]=prGG2Vz(GG2, PRSSinfo);
[Vz0,Vx0,pVz0,Ms0,Mf0,R0, GGf0]=GG2vUS(GG2, g1Vz0, Ms0, MfR0, PRSSinfo);
% Vz0=gather(Vz0.*CR0); Vx0=gather(Vx0.*CR0); % mm/s
% pVz0=gather(pVz0.*CR0); Ms0=gather(Ms0.*CR0); Mf0=gather(Mf0.*CR0); R0=gather(R0); GGf0=gather(GGf0);
Vz0=gather(Vz0); Vx0=gather(Vx0); % mm/s
pVz0=gather(pVz0); Ms0=gather(Ms0); Mf0=gather(Mf0); R0=gather(R0); GGf0=gather(GGf0);
if PRSSinfo.rfnScale>1
    Ms(:,:)=imresize(Ms0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    Mf(:,:)=imresize(Mf0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    Vx(:,:)=imresize(Vx0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    Vz(:,:)=imresize(Vz0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    pVz(:,:)=imresize(pVz0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    R(:,:)=imresize(R0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    CR(:,:)=gather(imresize(CR0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest')); % spatial interpolation
    for iTau=1:nTau
        GGf(:,:,iTau)=imresize(GGf0(:,:,iTau),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    end
else
    Ms=Ms0;  Mf=Mf0;  R=R0;  CR=gather(CR0);
    Vx=Vx0;  Vz=Vz0;  pVz=pVz0;  
    GGf(:,:,:)=GGf0;
end
V=sqrt(Vx.^2+Vz.^2).*sign(Vz);

% %% V. vUS fitting
% % V.1. Fitting constraint
% [Vx0,Vz0,PVz0,MfI0,R0]=iniVx0Vz0Pvz0(GG2, g1Vz0, Ms0, MfR0, PRSSinfo);
% sVz0=sign(Vz0);
% sVz0(sVz0<1)=0;
% Fmin_cstrn(:,:,1)=[Ms0-0.1 Ms0+0.1];   % Ms constrain
% Fmin_cstrn(:,:,2)=[max(MfR0-0.15, 0) min(MfR0+0.1,1-Ms0)];   % MfR constrain
% Fmin_cstrn(:,:,3)=[max(MfI0-0.15, 0) min(MfI0+0.2,1)];   % MfI constrain
% Fmin_cstrn(:,:,4)=[0.5*Vx0 1.3*Vx0+0.2e-3]*tau(end)/(Sigma2(1)); % Vx constrain
% Fmin_cstrn(:,:,5)=sign(Vz0).*[sVz0.*0.6.*abs(Vz0)+(1-sVz0).*1.2.*abs(Vz0),(1-sVz0).*0.6.*abs(Vz0)+sVz0.*1.2.*abs(Vz0)];  % Vz constrain, mm/s
% Fmin_cstrn(:,:,6)=[0.8*PVz0 min(PVz0*1.3,0.7)]; % PVz constrain
% fitC0(:,:,1) = double(Ms0); % initials
% fitC0(:,:,2) = double(MfR0); % initials
% fitC0(:,:,3) = double(MfI0); % initials
% fitC0(:,:,4) = double(Vx0*tau(end)/(Sigma2(1))); % initials
% fitC0(:,:,5) = double(Vz0); % initials
% fitC0(:,:,6) = double(PVz0); % initials
% fitC0=double(gather(fitC0));
% Fmin_cstrn=double(gather(Fmin_cstrn));
% % GG2=gather(GG2);
% %% V.2 fit complex (g1)
% fitE = @(c) double(sum( abs(c(:,1,1) + c(:,1,2).*exp( -(c(:,1,4).*tn).^2-(c(:,1,5).*tau).^2/(Sigma2(3))^2).*exp(-(k0*tau.*c(:,1,5).*c(:,1,6)).^2).*cos(2*k0*c(:,1,5).*tau)+...
%     1i*c(:,1,3).*exp( -(c(:,1,4).*tn).^2-(c(:,1,5).*tau).^2/(Sigma2(3))^2).*exp(-(k0*tau.*c(:,1,5).*c(:,1,6)).^2).*sin(2*k0*c(:,1,5).*tau)- (GG2) ).^2 ,2));
% [fitC, fval] = fmincon(fitE, fitC0, [],[],[],[], ...
%     [Fmin_cstrn(:,1,1) Fmin_cstrn(:,1,2) Fmin_cstrn(:,1,3) Fmin_cstrn(:,1,4) Fmin_cstrn(:,1,5) Fmin_cstrn(:,1,6)], ...
%     [Fmin_cstrn(:,2,1) Fmin_cstrn(:,2,2) Fmin_cstrn(:,2,3) Fmin_cstrn(:,2,4) Fmin_cstrn(:,2,5) Fmin_cstrn(:,2,6)], ...
%     [], optimset('Display','off','TolFun',1e-6,'TolX',1e-6));%
% 
% Ms=reshape(fitC(:,1,1),[nz,nx]); 
% Mf(:,:,1)=reshape(fitC(:,1,2),[nz,nx]);  % MfR
% Mf(:,:,2)=reshape(fitC(:,1,3),[nz,nx]);  % MfI
% Vx=reshape(fitC(:,1,4),[nz,nx])/(tau(end)/(Sigma2(1))).*CR; % m/s
% Vz=reshape(fitC(:,1,5),[nz,nx]).*CR;  % m/s
% V=sqrt(Vz.^2+Vx.^2);
% pVz=reshape(fitC(:,1,6),[nz,nx]).*CR; % 
% GGf0=fitC(:,:,1) + fitC(:,:,2).*exp( -(fitC(:,:,4).*tn).^2-(fitC(:,:,5).*tau).^2/(Sigma2(3))^2).*exp(-(k0*tau.*fitC(:,:,5).*fitC(:,:,6)).^2).*cos(2*k0*fitC(:,:,5).*tau)+...
%     1i*fitC(:,:,3).*exp( -(fitC(:,:,4).*tn).^2-(fitC(:,:,5).*tau).^2/(Sigma2(3))^2).*exp(-(k0*tau.*fitC(:,:,5).*fitC(:,:,6)).^2).*sin(2*k0*fitC(:,:,5).*tau);
% R=gather((reshape((1-sum(abs(GG2-GGf0).^2,2)./sum(abs((GG2)-mean(GG2,2)).^2,2)),[nz,nx]).*CR));  
% GGf=gather(reshape(GGf0,[nz,nx,nTau]));
% V=gather(V*1e3); Vz=gather(Vz*1e3); Vx=gather(Vx*1e3) ;% mm/s
% Mf=gather(Mf); Ms=gather(Ms); pVz=gather(pVz);            
