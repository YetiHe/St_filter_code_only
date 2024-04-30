%% US g1 fit, SV model, CPU
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
function [Mf, Vx, Vz, V, pVz ,R, Ms, CR, GGf]=sIQ2vUS_SV(sIQ, PRSSinfo)
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
%% I. signal-to-noise ratio of sIQ
[nz0,nx0,nt]=size(sIQ);
fCoor=linspace(-PRSSinfo.rFrame/2,PRSSinfo.rFrame/2,nt)';
fCoorSig=zeros(size(fCoor));
fCoorSig(abs(fCoor)<800)=1; % signal frequency range
fCoorSig=circshift(fCoorSig,nt/2);
fIQ=(fft(sIQ,nt,3)); % no fft shift
SNR0=squeeze(sum(abs(fIQ.*repmat(permute(fCoorSig,[3 2 1]),[nz0 nx0 1])),3))./squeeze(sum(abs(fIQ),3)); % SNR of oringla data
clear fIQ; 
%% II. sIQ2GG and spatial refine GG and SNR
PRSSinfo.g1StartT=1;
GG =sIQ2GG(sIQ, PRSSinfo);
clear sIQ;
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
[Vz0,Vx0,pVz0,Ms0,Mf0,R0, GGf0]=GG2vUS(GG2, g1Vz0, Ms0, MfR0, PRSSinfo);
Vz0=(Vz0.*CR0); Vx0=(Vx0.*CR0); % mm/s
pVz0=(pVz0.*CR0); Ms0=(Ms0.*CR0); Mf0=(Mf0.*CR0); R0=(R0); GGf0=(GGf0);
if PRSSinfo.rfnScale>1
    Ms(:,:)=imresize(Ms0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    Mf(:,:)=imresize(Mf0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    Vx(:,:)=imresize(Vx0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    Vz(:,:)=imresize(Vz0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    pVz(:,:)=imresize(pVz0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    R(:,:)=imresize(R0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    CR(:,:)=imresize(CR0(:,:),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    for iTau=1:nTau
        GGf(:,:,iTau)=imresize(GGf0(:,:,iTau),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
    end
else
    Ms=Ms0;  Mf=Mf0;  R=R0;  CR=CR0;
    Vx=Vx0;  Vz=Vz0;  pVz=pVz0;  
    GGf(:,:,:)=GGf0;
end
V=sqrt(Vx.^2+Vz.^2).*sign(Vz);
