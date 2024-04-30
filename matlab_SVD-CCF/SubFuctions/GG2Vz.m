%% g1 based Vz calcualtion, CPU
% formula: 2pi/T=2n*2pi/Lambda*Vz => Vz=Lambda/(2nT), n is the optical refractive index
% input: 
    % GG, 2D array, (nVox,nTau), nVox=nz*nx
    % PRSSinfo: data processing parameters, including 
        % PRSSinfo.rFrame: sIQ frame rate, Hz
        % PRSSinfo.f0: Transducer center frequency, Hz
        % PRSSinfo.C: Sound speed in the sample, m/s
        % PRSSinfo.Dim: GG original dimension, [nz,nx,nTau]
        % PRSSinfo.rfnScale: spatial refind scale
% output: 
    % Vz, 1D, [nVox,1], m/s
%%%%%%%%%%%%%%% EXAMPLE %%%%%%%%%%%%%%%%%%%%%%

function [Vz, Tvz]=GG2Vz(GG, PRSSinfo, nItp)
if nargin<3
    nItp=10;
end
lambda=PRSSinfo.C/PRSSinfo.f0;        % wavlength, m
rFrameItp=PRSSinfo.rFrame*nItp;       % frame rate after temporal interpolation
nz=PRSSinfo.Dim(1); nx=PRSSinfo.Dim(2); 
%% I. temporal resampling
[nVox,nTau]=size(GG); % nVox=nz*nx*ny
sTau=linspace(1,nTau,nTau).';
rsTau=linspace(1,nTau,nTau*nItp).';
rGG=movmean(interp1(sTau,real(GG).',rsTau,'linear'),50,1); %[nVox,nTau]
iGG=movmean(interp1(sTau,imag(GG).',rsTau,'linear'),50,1); %[nVox,nTau]

%% Vz calculation
% RRHP=conv2(HPknl,1, RR2,'same');
% RRUnwrap=movmean(unwrap(angle(iGG),[],2),5,2);
% ST=(RRUnwrap(:,end)-RRUnwrap(:,1))/nTau*1/2; % Slope Threshold based on the difference of the last and fisrt unwraped phase
% PS=diff(RRUnwrap,1,2); % Phase Difference of the slope
% sRBC=bsxfun(@gt,PS.*sign(ST),ST.*sign(ST)*1.2); % slope greater than the slope threshold, identified as RBC flowing through


%% II. find the period
%% approx decay time
% GGtd=sign(max(min(abs(GG),[],2),0.2)-abs(GG));
% [~,Ind]=max(GGtd,[],2);
% Tdc=(Ind/PRSSinfo.rFrame).';
%% Tvz obtained from the first Valley of rGG
dGG=(sign(diff(rGG,1,1))==1);
[vGG,incsGG]=max(dGG,[],1);
HalfCyc=incsGG(1,:);
HalfCyc(HalfCyc<5)=nTau*nItp;
HalfCyc(vGG==0)=nTau*nItp;
Tvz=HalfCyc/rFrameItp*2; % period, s
% ggCR=((abs(GG(:,1))>0.1));
% iggCR=(max(abs(iGG(1:10*nItp,:)),[],1)>=0.15).';
% CR=ggCR.*iggCR; % threshold criteria
% CR=CR+(1-CR).*abs(GG(:,1))*2;
%% Tvz obtained with performing autocorrelation on rGG
% ACF=aCorr((movmean(rGG',5,2))',nTau*floor(nItp/2), 1);
% diffACF=(sign(diff(ACF,1,1))==1);
% [vACF,incsACF]=max(diffACF,[],1);
% HalfCyc=incsACF(1,:);
% HalfCyc(vACF==0)=nTau*nItp;
% Tvz_acf=HalfCyc/rFrameItp*2; % period, s

ACF=aCorr((movmean(real(GG),5,2))',nTau, 1);
diffACF=(sign(diff(ACF,1,1))==1);
[vACF,incsACF]=max(diffACF,[],1);
HalfCyc=incsACF(1,:);
HalfCyc(vACF==0)=nTau*2;
Tvz_acf=HalfCyc/PRSSinfo.rFrame*2; % period, s
% Tvz
ddGG=diff(dGG,1,1);
ddGG(ddGG<1)=0;
NdGG=sum(ddGG,1);
Tvz=Tvz.*(NdGG<5)+Tvz_acf.*(NdGG>=5);
% % Tvz(Tvz<Tdc/5)=Tdc(Tvz<Tdc/5);
%% III. Vz
% Vz0=(((lambda./(2*Tvz).*sign(mean(ST(1:3,:),1))).')); % absolute value of axial velocity, m/s
Vz0=(((lambda./(2*Tvz).*sign(mean(iGG(1:3,:),1))).')); % absolute value of axial velocity, m/s
Vz=reshape(imgaussfilt(reshape(Vz0,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),0.8),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
