%% US g1 fit for in vivo data, fit negative and postive frequency signal separately, CPU
% input: 
    % sIQ: bulk motion removed data, [nz,nx,nt]
    % PRSSinfo: data processing parameters, including 
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
        % PRSSinfo.MpVz: maximu pVz
        % PRSSinfo.useMsk: 1: use ULM data as spatial mask; 0: no spatial mask
        % PRSSinfo.ulmMsk: ULM-based spatial constrain mask
            % [nz,nx,3], 1: up flow (positive frequency); 2 down flow (negative
            % frequency); 3: all flow 
            % ulmMsk=1 otherwise
 % output:
    % Ms: static component fraction, [nz,nx]
    % Mf: dynamic component fraction, [nz,nx,2], 2: [real,imag]
    % Vx: x-direction velocity component, [nz,nx], mm/s
    % Vz: axial-direction velocity component, [nz,nx], mm/s
    % V=sqrt(Vx.^2+Vz.^2), [nz,nx], mm/s
    % pVz: Vz distribution (sigma-Vz), [nz,nx]
    % R: fitting accuracy, [nz,nx]
    % CR: freqCR.*pnCR.*MskCR.*ggCR
    % GGf: gg fitting results, [nz,nx, nTau]
 % Jianbo Tang, 20190820
function [Mf, Vz, V, pVz, Vcz, R, CR, Vx, Ms, pnRatio,GGf]=sIQ2vUS_NPDV(sIQ, PRSSinfo)
%% O. constant
lambda0=PRSSinfo.C/PRSSinfo.f0;        % wavlength
k0 = 2*pi/lambda0;   % wave number
PRSSinfo.FWHM=[PRSSinfo.FWHM(1) 1000e-6 PRSSinfo.FWHM(2)]; % just to put the FWHM_y, any number other than 0
Sigma=PRSSinfo.FWHM*0.7/(2*sqrt(2*log(2))); % intensity-based sigma
Sigma2=2*Sigma;
nItpVz0=10;          % for Vz0 determination
dt = 1/PRSSinfo.rFrame;      % frame interval, s
tau = [1:PRSSinfo.g1nTau]*dt; % time lag, s
nTau0=PRSSinfo.g1nTau;
tn = tau / tau(end);
fRangeSignal=1000; % signal frequency range, Hz
%% I. determine spectrum power ratio and signal (|f|<1000Hz) to noise ratio
[nz0,nx0,nt]=size(sIQ);
% I.0 all frequency signal and SNR
PRSSinfo.g1StartT=1;
PRSSinfo.g1nTau=3;
GG0 = sIQ2GG(sIQ, PRSSinfo); % g1 of whole frequency signal
PRSSinfo.g1nTau=nTau0;
fCoor=linspace(-PRSSinfo.rFrame/2,PRSSinfo.rFrame/2,nt)';
fCoorSig=zeros(size(fCoor));
fCoorSig(abs(fCoor)<1100)=1; % signal frequency range
fCoorSig=circshift(fCoorSig,nt/2);
% fM=ones(1,1,nt);
% HfRange=find(abs(fCoor)<(PRSSinfo.rFrame/2-fRangeSignal)==1);
% % fM(1,1,HfRange)=((HfRange-(HfRange(floor(end/2)))).^2+0.2)/((HfRange(1)-(HfRange(floor(end/2)))).^2+0.2);
% fM(1,1,HfRange(floor(end/2)+1:end))=((HfRange(floor(end/2)+1:end)-(HfRange(floor(end/2)))).^2+0.3)/((HfRange(1)-(HfRange(floor(end/2)))).^2+0.3);
% fIQ=(fft(sIQ,nt,3)).*fM; % no fft shift
fIQ=sysNoiseRemove(sIQ,PRSSinfo.rFrame);

% I.1 frequency-based SNR
zPix=max(floor(nz0*0.1),1):1:floor(nz0*1);
fSNR0=squeeze(sum(abs(fIQ.*repmat(permute(fCoorSig,[3 2 1]),[nz0 nx0 1])),3))./squeeze(sum(abs(fIQ),3)); % SNR of original data
zfSNR=PRSSinfo.useMsk*(mean(fSNR0,2)-1.3*std(fSNR0,[],2))+(1-PRSSinfo.useMsk)*(mean(fSNR0,2)-0.9*(1+([1:nz0]./(5*nz0)).^2)'.*std(fSNR0,[],2));
fC=polyfit(zPix,zfSNR(max(floor(nz0*0.1),1):floor(nz0*1))',1);
fSNRthd0=repmat(polyval(fC,[1:nz0])',[1, nx0])*1.02;     
% I.2 all frequency spatial mask
if PRSSinfo.useMsk==1
    MskCR=(abs(PRSSinfo.ulmMsk(:,:,3))>0.5);
else
    MskCR=1;
end
% I.3 binary mask - all frequency
gR=(mean(real(GG0(:,:,1:2)),3));
gRm=mean(gR(:));
gRstd=std(gR(:));
% aCR=(((fSNR0>fSNRthd0)+abs(GG0(:,:,1))>0.4)>0).*...
%     (PRSSinfo.useMsk*(gR>max((gRm-0.7*gRstd),0.03))+(1-PRSSinfo.useMsk)*(gR>max((gRm-0.4*gRstd),0.08))); % all frequency signal-based thresholding
aCR=(((fSNR0>fSNRthd0)+abs(GG0(:,:,1))>0.4)>0).*...
    (PRSSinfo.useMsk*1+(1-PRSSinfo.useMsk)*(gR>max((gRm-0.4*gRstd),0.08))); % all frequency signal-based thresholding
clear GG0 sIQ
%% II. positive&negative frequncy signal vUS processing
% PRSSinfo.MpVz=0.8; % maximu pVz
DispPrss={'Positive frequency signal', 'Negative frequency signal'};
for iNP=1:2
    iFIQ=zeros(size(fIQ));
    iFIQ(:,:,(floor(nt/2))*(iNP-1)+1:floor(nt/2)*iNP)=fIQ(:,:,(floor(nt/2))*(iNP-1)+1:floor(nt/2)*iNP);
    %% III. acceptable signal criteria, CR
    % III. 1 freqCR
    iFIQ_S=iFIQ.*repmat(permute(fCoorSig,[3 2 1]),[nz0 nx0 1]);
    fSNR0=squeeze(sum(abs(iFIQ_S),3))./squeeze(sum(abs(iFIQ),3)); % SNR of frequency signal
    zfSNR=PRSSinfo.useMsk*(mean(fSNR0,2)-1.0*std(fSNR0,[],2))+(1-PRSSinfo.useMsk)*(mean(fSNR0,2)-0.8*(1+([1:nz0]./(5*nz0)).^2)'.*std(fSNR0,[],2));
    fC=polyfit(zPix,zfSNR(max(floor(nz0*0.1),1):floor(nz0*1))',1);
%     fSNRthd0=repmat(polyval(fC,[1:nz0])',[1, nx0])*(1+(iNP-1)*0.1);
    fSNRthd0=repmat(polyval(fC,[1:nz0])',[1, nx0])*(1.05);
    fCR=(fSNR0>fSNRthd0);
%     fCR=fCR+(1-fCR).*fSNR0*0.5;
    fCR0=(fSNR0>0.6);
    % III. 2 pnCR
    ipnRatio=sum(abs(iFIQ(:,:,(floor(nt/2))*(iNP-1)+1:floor(nt/2)*iNP))-repmat(median(abs(fIQ(:,:,:)),3),[1 1 floor(nt/2)]),3); % spectrum power of positive frequency
    ipnRatio=ipnRatio./(sum(abs(fIQ(:,:,:))-repmat(median(abs(fIQ(:,:,:)),3),[1 1 nt]),3));
    pnCR=(ipnRatio>(PRSSinfo.useMsk*(0.15+(0.05*(iNP-1)))+(1-PRSSinfo.useMsk)*(0.2+(0.1*(iNP-1)))));
    % III. 3 iGG and ggCR
    iIQ=(ifft(iFIQ,nt,3));
    iGG=sIQ2GG(iIQ, PRSSinfo); % g1 of p or n frequency signal
    gR=(mean(abs(iGG(:,:,1:3)),3));
    gRm=mean(gR(:));
    gRstd=std(gR(:));
    ggCR=(PRSSinfo.useMsk*(gR>max((gRm-0.6*gRstd),0.15))+(1-PRSSinfo.useMsk)*(gR>max((gRm-0.4*gRstd),0.25)));
    ggCR0=((abs(iGG(:,:,1)))>0.6);
    % III. 4 CR for positive or negative frequency signal
    iCR=(((pnCR+fCR0)>0).*((fCR+ggCR0)>0).*ggCR.*aCR);  % acceptable signal criteria
%     iCR=(((((pnCR+fCR0)>0).*fCR.*ggCR.*aCR)+mean(abs(iGG(:,:,1:3)),3)>0.4)>0);  % acceptable signal criteria
    %% IV Color Doppler
    iVcz=(ColorDoppler_NP(iIQ,PRSSinfo)); % color Doppler
    clear iIQ iFIQ ggCR pnCR fCR0 fCR
    %% V. vUS fitting
    [nz0,nx0,nTau]=size(iGG);
    if nz0*nx0==1
        iCR=1;
    end
    % V.1. GG2Vz
    PRSSinfo.Dim=[nz0,nx0,nTau];
%     [g1Vz0]=GG2Vz_FREQ(iGG, PRSSinfo);
    % V.2. vUS initial and fitting
    B=ones(3,3)/9;
    for iTau=1:nTau
        iGG(:,:,iTau)=convn(iGG(:,:,iTau),B,'same');
    end
    GG2=reshape(iGG,[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),PRSSinfo.Dim(3)]);
    clear iGG
    % V.3. g1(1) adjust
    ggCR1=(abs(abs(GG2(:,1))-abs(GG2(:,2)))>2*abs((abs(GG2(:,2))-abs(GG2(:,3)))));
    ggCR2=(abs(GG2(:,1))>0.55).*(abs(GG2(:,2))<0.25).*(abs(GG2(:,2))<abs(GG2(:,3)));
    ggCR0=((ggCR1+ggCR2)>0);
    GG2(:,1)=(1-ggCR0).*GG2(:,1)+ggCR0.*(GG2(:,2)+(abs(real(GG2(:,2)-GG2(:,3)))+1i*abs(imag(GG2(:,2)-GG2(:,3))))*1.5);
    
    Ms00 = min(max(real(FindCOR(GG2(:,floor(end*1/2):end))),0),max(mean(real(GG2(:,floor(end*2/3):end)),2),0));
    Me0 =1-abs(GG2(:,1));  
%     MfR0 = max(min(1-Ms00-Me0,real(GG2(:,1))),0);
    MfR0 = max(min(1-Ms00-Me0,1),0);
    [g1Vz0, Tvz]=GG2Vz(GG2, PRSSinfo, 10);
    [iVz, iVx, ipVz, iMs, iMf, iR, iGGf]=GG2vUS(GG2, reshape(g1Vz0,[nz0*nx0,1]), Ms00, MfR0, PRSSinfo);
%     [iVz, iVx, ipVz, iMs, iMf, iR, iGGf]=GG2vUS(GG2, reshape(iVcz,[nz0*nx0,1]), Ms00, MfR0, PRSSinfo);
    Ms0(:,:,iNP)=(iMs); Mf0(:,:,iNP)=(iMf(:,:,1));
    Vx0(:,:,iNP)=(iVx).*iCR; Vz0(:,:,iNP)=(iVz).*iCR; Vcz0(:,:,iNP)=-1*iVcz.*iCR;
    pVz0(:,:,iNP)=(ipVz).*iCR; R0(:,:,iNP)=(iR); GGf0(:,:,:,iNP)=(iGGf);
    CR0(:,:,iNP)=(iCR);
    pnRatio0(:,:,iNP)=(ipnRatio);
end
Vz0(:,:,1)=-1*abs(Vz0(:,:,1));
Vz0(:,:,2)=1*abs(Vz0(:,:,2));
% Vx0(abs(Vx0)>40)=20;
% III. 3 MaskCR
if PRSSinfo.rfnScale>1
    for iNP=1:2
        if PRSSinfo.useMsk==1
            mskCR=PRSSinfo.ulmMsk(:,:,iNP);
        else
            mskCR=1;
        end
        Ms(:,:,iNP)=imresize(Ms0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
        Mf(:,:,iNP)=imresize(Mf0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
        Vx(:,:,iNP)=imresize(Vx0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest').*mskCR; % spatial interpolation
        Vz(:,:,iNP)=imresize(Vz0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest').*mskCR; % spatial interpolation
        Vcz(:,:,iNP)=imresize(Vcz0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest').*mskCR; % spatial interpolation
        pVz(:,:,iNP)=imresize(pVz0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest').*mskCR; % spatial interpolation
        R(:,:,iNP)=imresize(R0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest'); % spatial interpolation
        CR(:,:,iNP)=imresize(CR0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest').*mskCR; % spatial interpolation
        pnRatio(:,:,iNP)=imresize(pnRatio0(:,:,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest').*mskCR; % spatial interpolation
        for iTau=1:nTau
            GGf(:,:,iTau,iNP)=imresize(GGf0(:,:,iTau,iNP),[nz0,nx0]*PRSSinfo.rfnScale,'nearest').*mskCR; % spatial interpolation
        end
    end
    GGf(:,:,:,3)=GGf(:,:,:,1).*repmat(pnRatio(:,:,1),[1,1,PRSSinfo.g1nTau])+GGf(:,:,:,2).*repmat(pnRatio(:,:,2),[1,1,PRSSinfo.g1nTau]);
else
    Ms=Ms0;  Mf=Mf0;  R=R0;  CR=CR0;
    Vx=Vx0;  Vz=Vz0;  pVz=pVz0;  pnRatio=pnRatio0;
    Vcz=Vcz0;
    GGf(:,:,:,1:2)=GGf0; 
    GGf(:,:,:,3)=GGf(:,:,:,1).*repmat(pnRatio(:,:,1),[1,1,PRSSinfo.g1nTau])+GGf(:,:,:,2).*repmat(pnRatio(:,:,2),[1,1,PRSSinfo.g1nTau]);
end
V=sqrt(Vx.^2+Vz.^2).*sign(Vz);
Vcz=Vcz*1e3; % mm/s
