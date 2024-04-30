%% vUS fitting
% Input:
    % GG, 2D array, (nVox,nTau), nVox=nz*nx
    % Vz0: 1D array, [nVox,1], initial value, m/s
    % Ms0: 1D array, [nVox,1], initial value for Ms
    % MfR0: 1D array, [nVox,1], initial value for Mf
    % PRSSinfo.C: sound speed, m/s
    % PRSSinfo.FWHM: (X, Y, Z) spatial resolution, Full Width at Half Maximum of point spread function, m
    % PRSSinfo.rFrame: sIQ frame rate, Hz
    % PRSSinfo.f0: Transducer center frequency, Hz
    % PRSSinfo.MpVz: maximu pVz
% output:
    % Ms: static component fraction, [nz,nx]
    % Mf: dynamic component fraction, [nz,nx,2], 2: [real,imag]
    % Vx: x-direction velocity component, [nz,nx], mm/s
    % Vz: axial-direction velocity component, [nz,nx], mm/s
    % V=sqrt(Vx.^2+Vz.^2), [nz,nx], mm/s
    % R: fitting accuracy, [nz,nx]
    % GGf: GG fitting results, [nz,nx, nTau]
% Jianbo Tang, 20190812
function [Vz,Vx,pVz,Ms,Mf,R,GGf]=GG2vUS(GG, Vz0, Ms0, MfR0, PRSSinfo)
%% I. DAQ parameter
[nVox,nTau]=size(GG);
tau=[1:nTau]/PRSSinfo.rFrame; % time lag, s
Sigma=PRSSinfo.FWHM*0.7/(2*sqrt(2*log(2))); % intensity-based sigma
Sigma2(3)=2*Sigma(3);
Sigma2(1)=0.7*Sigma(1);
lambda0=PRSSinfo.C/PRSSinfo.f0;        % wavlength
k0 = 2*pi/lambda0;   % wave number

%% II. Vx, Vz and PVz mesh
if nVox~=1
    Vz0=reshape(medfilt2(reshape(Vz0,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),[7,7]),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
    MfR00=reshape(medfilt2(reshape(MfR0,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),[7,7]),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
else
    MfR00=MfR0;
end

if PRSSinfo.MpVz==0
    % Vx0 initial grid
    NmVx=400;
    mVx0=linspace(0,1,NmVx);
    [mVx,mVx00,~]=ndgrid((800e-3),mVx0,tau);
    NmPVz=1;
    mPVz=0; StepPvz=0;
else
    % Vx0 initial grid
    NmVx=8;
    mVx0=linspace(0,1,NmVx);
%     sVmsk=(Vz0>-4e-3).*(Vz0<4e-3);
%     VxMax1=max(abs(Vz0)*15,5);                      % when -5<Vz0<3 mm/s
%     VxMax2=min(250./abs(Vz0*1e3),20)*1e-3;  % when Vz0>3 or <-5 mm/s
%     VxMax=((VxMax1.*sVmsk)+VxMax2.*(1-sVmsk)).*MfR0.^0.2;

%     sVmsk=(abs(Vz0)<10e-3);
%     VxMax1=max(min(100./(abs(Vz0*1e3)),20)*1e-3.*MfR0.^0.15,5e-3);  
%     VxMax2=max(min(200./(abs(Vz0*1e3)),20)*1e-3.*MfR0.^0.15,5e-3);  
%     VxMax=((VxMax1.*sVmsk)+VxMax2.*(1-sVmsk)).*MfR0.^0.2;

%     VxMax=max(min(160./(abs(Vz0*1e3)),20)*1e-3.*MfR0.^0.11,5e-3);  % when Vz0>3 or <-5 mm/s
    
%     VxMax=(max(13*1e-3.*MfR0.^0.1,8e-3).*(MfR0>0.2)+(MfR0<=0.2)*5e-3).*min((abs(Vz0)/3e-3),1);  
%     VxMax=max(15*1e-3.*MfR0.^0.1,8e-3).*min((abs(Vz0)/3e-3),1);
    gR=mean(abs(GG(:,1:2)),2);
    gCR=(gR>0.35)*1+(gR<=0.35).*gR*3;
    gCR=reshape(imgaussfilt(reshape(gCR,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),1),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
    VxMax=max(18*1e-3.*MfR00.^0.2,8e-3).*(PRSSinfo.useMsk+(1-PRSSinfo.useMsk)*max(min((abs(Vz0)/3e-3),1),0.2)).*(gCR);
    [mVx,mVx00,~]=ndgrid(VxMax,mVx0,tau);
    % pVz0 initial grid
%     NmPVz=7;
%     mPVz0=linspace(0.3,PRSSinfo.MpVz,NmPVz);
%     mPVz=repmat(mPVz0,[nVox,1,nTau]);
%     StepPvz=mPVz(:,2,1)-mPVz(:,1,1);
        
    NmPVz=7;
    mPVz0=linspace(1,3,NmPVz);
    if nVox~=1
        mPVzMin=(abs(Vz0)>5e-3)*0.33+(abs(Vz0)<=5e-3)*0.2;
        %      mPVzMin=(abs(Vz0)>5e-3).*(0.5+min(abs(Vz0)/15e-3,0.15))+(abs(Vz0)<=5e-3).*(0.3-min(0.1*5e-3./abs(Vz0),0.15));
        mPVzMin=reshape(medfilt2(reshape(mPVzMin,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),[5,5]),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
        %     mPVzMin=reshape(imgaussfilt(reshape(mPVzMin,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),1),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
    else
        mPVzMin=0.3;
    end
    [mPVz,mPVz00,~]=ndgrid(mPVzMin,mPVz0,tau);
    mPVz=mPVz.*mPVz00;
    StepPvz=mPVz(:,2,1)-mPVz(:,1,1);
end


mVx=mVx.*mVx00;
StepVx=mVx(:,2,1)-mVx(:,1,1);
clear mVx00 mPVz00 mPVzMax
[mMs, ~]=ndgrid(Ms0,tau);
[mMf, Tau]=ndgrid(MfR0,tau);

%% real part of g1 
NmVz=200;
mVz0=linspace(0.7, 1.3, NmVz);                      
[mVz,mVz00]=ndgrid(Vz0,mVz0);
mVz1=mVz.*mVz00;
clear mVz00;
StepVz=mVz1(:,2)-mVz1(:,1);
if isa(GG,'gpuArray')
    RR=zeros(nVox,NmVz,NmVx,NmPVz,'gpuArray');
else
    RR=zeros(nVox,NmVz,NmVx,NmPVz);
end
for iVz=1:NmVz
    iVz0=mVz1(:,iVz);
    [mVz, ~]=ndgrid(iVz0,tau);
    for iVx=1:NmVx
        for iPvz=1:NmPVz
            GGr(:,:,iPvz)=mMs + mMf.*exp(-permute(mVx(:,iVx,:),[1,3,2]).^2.*Tau.^2/(Sigma2(1))^2-mVz.^2.*Tau.^2/(Sigma2(3))^2)...
                .*exp(-(k0*Tau.*permute(mPVz(:,iPvz,:),[1,3,2]).*mVz).^2).*cos(2*k0*mVz.*Tau);
        end
        RR(:,iVz,iVx,:)=1 - permute((sum( abs(repmat(real(GG),[1,1,NmPVz])-GGr).^2,2)...
            ./ repmat(sum( abs(real(GG)-mean(real(GG),2)).^2,2),[1,1,NmPVz])),[1,2,4,3]);
    end
end
clear GGr;
[R0,InD]=max(RR(:,:),[],2);
[MIvz, MIvx, MIPvz]=ind2sub([NmVz, NmVx,NmPVz],InD);
Vz0=mVz1(:,1)+(MIvz-1).*StepVz; % m/s
% Vx00=(mVx(:,1,1)+(MIvx-1).*StepVx).*(R0>0)+((80e-3./abs(Vz0))*1e-3.*MfR0).*(R0<=0); % m/s
Vx00=(mVx(:,1,1)+(MIvx-1).*StepVx);
Vx00(isnan(abs(Vx00)))=0;
B=ones(5,5)/25;
pVz00(:,1)=(mPVz(:,1,1)+(MIPvz-1).*StepPvz);
if nVox~=1
    Vx0=reshape(convn(reshape(Vx00,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),B,'same'),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
    pVz0=reshape(convn(reshape(pVz00,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),B,'same'),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
else
    Vx0=Vx00;
    pVz0=pVz00;
end
% Vx0=reshape(medfilt2(reshape(Vx00,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),[7,7]),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
% Vx0=reshape(imgaussfilt(reshape(Vx00,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),1),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
% Vx0=min((mVx(:,1,1)+(MIvx-1).*StepVx),(200e-3./abs(Vz0))*1e-3.*MfR0);
% pVz0=reshape(medfilt2(reshape(pVz00,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),[5,5]),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
% pVz0=reshape(imgaussfilt(reshape(pVz00,[PRSSinfo.Dim(1),PRSSinfo.Dim(2)]),0.8),[PRSSinfo.Dim(1)*PRSSinfo.Dim(2),1]);
%% III. determine MfI0 from the imag GG
NmMfI=100;
mMfI0=linspace(1.1,1.5, NmMfI);
[mMfI,mMfI00]=ndgrid(max(abs(imag(GG)),[],2),mMfI0);
mMfI1=mMfI.*mMfI00;
StepMfI=mMfI1(:,2)-mMfI1(:,1);
for imMfI=1:NmMfI
    GGi(:,:,imMfI)=mMfI1(:,imMfI).*exp(-Vx0.^2.*tau.^2/(Sigma2(1))^2-Vz0.^2.*tau.^2/(Sigma2(3))^2).*exp(-(k0*tau.*pVz0.*Vz0).^2).*sin(2*k0*Vz0.*tau);
end
RRi=1 - (sum( abs(repmat(imag(GG),[1,1,NmMfI])-GGi).^2,2))...
    ./ repmat(sum( abs(imag(GG)-mean(imag(GG),2)).^2,2),[1,1,NmMfI]);
[RI0,MindMfI]=max(RRi,[],3);
MindMfI0=MindMfI(:,1);
MfI0=min(mMfI1(:,1)+(MindMfI0-1).*StepMfI,1); % m/s
clear mVx mVx00 mPvz mMs mMf mMfI mMfI00 GGi RR
%% IV. g1 fit, finer grid
NmVx=200;
mVx0=linspace(0.8, 1.2, NmVx);
[mVx,mVx00,~]=ndgrid(Vx0,mVx0,tau);
mVx=mVx.*mVx00;
StepVx=mVx(:,2,1)-mVx(:,1,1);

NmPVz=100;
mpVz0=linspace(0.75,1.15,NmPVz);
[mpVz,mpVz00,~]=ndgrid(pVz0,mpVz0,tau);
mpVz=mpVz.*mpVz00;
StepPvz=mpVz(:,2,1)-mpVz(:,1,1);

[mMs, ~]=ndgrid(Ms0,tau);
[mMfR, ~]=ndgrid(MfR0,tau);
[mMfI, ~]=ndgrid(MfI0,tau);
[mVz, Tau]=ndgrid(Vz0,tau);
%% complex part of g1 
if isa(GG,'gpuArray')
    RR=zeros(nVox,NmVx,NmPVz,'gpuArray');
else
    RR=zeros(nVox,NmVx,NmPVz);
end
for iVx=1:NmVx
    for iPvz=1:NmPVz
        GGc(:,:,iPvz)=mMs + exp(-permute(mVx(:,iVx,:),[1,3,2]).^2.*Tau.^2/(Sigma2(1))^2-mVz.^2.*Tau.^2/(Sigma2(3))^2)...
            .*exp(-(k0*Tau.*permute(mpVz(:,iPvz,:),[1,3,2]).*mVz).^2).*(mMfR.*cos(2*k0*mVz.*Tau)+mMfI.*1i.*sin(2*k0*mVz.*Tau));
    end
    RR(:,iVx,:)=1.2 - permute((sum( abs(repmat((GG),[1,1,NmPVz])-GGc).^2,2)...
    ./ repmat(sum( abs((GG)-mean((GG),2)).^2,2),[1,1,NmPVz])),[1,2,3]);
end
clear GG GGc
[R,indR]=max(RR(:,:),[],2);
[MIvx, MIPvz]=ind2sub([NmVx,NmPVz],indR);
Vx=reshape((mVx(:,1,1)+(MIvx-1).*StepVx)*1e3, [PRSSinfo.Dim(1), PRSSinfo.Dim(2)]); % mm/s
pVz=reshape((mpVz(:,1,1)+(MIPvz-1).*StepPvz), [PRSSinfo.Dim(1), PRSSinfo.Dim(2)]); %
Vz=reshape(Vz0*1e3, [PRSSinfo.Dim(1), PRSSinfo.Dim(2)]); % mm/s
Ms=reshape(Ms0, [PRSSinfo.Dim(1), PRSSinfo.Dim(2)]);
R=reshape(R, [PRSSinfo.Dim(1), PRSSinfo.Dim(2)]);
Mf(:,:,1)=reshape(MfR0, [PRSSinfo.Dim(1), PRSSinfo.Dim(2)]);
Mf(:,:,2)=reshape(MfI0, [PRSSinfo.Dim(1), PRSSinfo.Dim(2)]);

[mVx1, ~]=ndgrid((mVx(:,1,1)+(MIvx-1).*StepVx),tau);
[mpVz1, ~]=ndgrid((mpVz(:,1,1)+(MIPvz-1).*StepPvz),tau);
GGf=mMs + exp(-mVx1.^2.*Tau.^2/(Sigma2(1))^2-mVz.^2.*Tau.^2/(Sigma2(3))^2)...
            .*exp(-(k0*Tau.*mpVz1.*mVz).^2).*(mMfR.*cos(2*k0*mVz.*Tau)+mMfI.*1i.*sin(2*k0*mVz.*Tau));
GGf=reshape(GGf,[PRSSinfo.Dim(1), PRSSinfo.Dim(2),PRSSinfo.Dim(3)]);


