%% remove the system noise in the frequency range of [450  600] Hz xxxxxxxxxxxxxxxxxxxxxx
function fIQ=sysNoiseRemove(IQ,rFrame)
[nz,nx,nt]=size(IQ);
fIQ=(fft(IQ,nt,3)); % Fourier transfer
fIQ=reshape(fIQ,[nz*nx,nt]);
fCoor=linspace(-rFrame/2,rFrame/2,nt)';
%% 1.  to eliminatet the system noise - negative frequency ([-1780 -800] Hz) 
fCoorSysNoiseN=zeros(size(fCoor));
fCoorSysNoiseN((fCoor)>-1780&(fCoor)<-500)=1; % system noise frequency range
fCoorSysNoiseN=circshift(fCoorSysNoiseN,nt/2);

fCoorSysNoiseRefN=zeros(size(fCoor));
fCoorSysNoiseRefN((fCoor)>-1780&(fCoor)<-1400)=1; % system noise frequency range
fCoorSysNoiseRefN=circshift(fCoorSysNoiseRefN,nt/2);

fCoorRefN=zeros(size(fCoor));
fCoorRefN((fCoor)>-1900&(fCoor)<-1800)=1; % reference frequency range
fCoorRefN=circshift(fCoorRefN,nt/2);
% find the noise pixel
fNoise0=fIQ(:,fCoorSysNoiseRefN>0);
mNoiseF0=mean(abs(fNoise0),2);
mRefF=mean(abs(fIQ(:,fCoorRefN>0)),2);
rN=mNoiseF0./mRefF;
rN=(rN>1.3);
% suppress the noise region
fNoise=fIQ(:,fCoorSysNoiseN>0);
mfNoise=movmean(abs(fNoise),10,2);
% MRefF=max(abs(fIQ(:,fCoorRefN>0)),[],2);
fIQ(:,fCoorSysNoiseN>0)=(1-rN).*fNoise+rN.*(fNoise./mfNoise.*mRefF);
fIQ=reshape(fIQ,[nz,nx,nt]);

% %% 2. to eliminatet the system noise - negative frequency ([-580 -421] Hz) 
% fCoorSysNoiseN=zeros(size(fCoor));
% fCoorRefN=zeros(size(fCoor));
% fCoorSysNoiseN((fCoor)>-580&(fCoor)<-421)=1; % system noise frequency range
% fCoorSysNoiseN=circshift(fCoorSysNoiseN,nt/2);
% fCoorRefN((fCoor)>-420&(fCoor)<-391)=1; % reference frequency range
% fCoorRefN=circshift(fCoorRefN,nt/2);
% nNoiseF=mean(abs(fIQ(:,:,fCoorSysNoiseN>0)),3);
% nRefF=mean(abs(fIQ(:,:,fCoorRefN>0)),3);
% rN=nNoiseF./nRefF;
% [IstrnNoiseN,JstrnNoiseN]=find(rN>1.5);
% for iI=1:length(IstrnNoiseN)
%     fCrctRef=mean(abs(squeeze(fIQ(IstrnNoiseN(iI), JstrnNoiseN(iI),fCoorRefN>0))));
%     fCrct=movmean(abs(squeeze(fIQ(IstrnNoiseN(iI), JstrnNoiseN(iI),fCoorSysNoiseN>0))),5)/fCrctRef;
%     fIQ(IstrnNoiseN(iI), JstrnNoiseN(iI),fCoorSysNoiseN>0)=squeeze(fIQ(IstrnNoiseN(iI), JstrnNoiseN(iI),fCoorSysNoiseN>0))./fCrct;
% end
% %% 3. to eliminatet the system noise - positive frequency ([451 610] Hz) 
% fCoorSysNoiseP=zeros(size(fCoor));
% fCoorRefP=zeros(size(fCoor));
% fCoorSysNoiseP((fCoor)>451&(fCoor)<610)=1; % system noise frequency range
% fCoorSysNoiseP=circshift(fCoorSysNoiseP,nt/2);
% fCoorRefP((fCoor)>421&(fCoor)<450)=1; % system noise frequency range
% fCoorRefP=circshift(fCoorRefP,nt/2);
% pNoiseF=mean(abs(fIQ(:,:,fCoorSysNoiseP>0)),3);
% pRefF=mean(abs(fIQ(:,:,fCoorRefP>0)),3);
% rP=pNoiseF./pRefF; % power ratio: noise frequency range to neighbor frequency range
% [IstrnNoiseP,JstrnNoiseP]=find(rP>1.5);
% for iI=1:length(IstrnNoiseP)
%     fCrctRef=mean(abs(squeeze(fIQ(IstrnNoiseP(iI), JstrnNoiseP(iI),fCoorRefP>0))));
%     fCrct=movmean(abs(squeeze(fIQ(IstrnNoiseP(iI), JstrnNoiseP(iI),fCoorSysNoiseP>0))),5)/fCrctRef;
%     fIQ(IstrnNoiseP(iI), JstrnNoiseP(iI),fCoorSysNoiseP>0)=squeeze(fIQ(IstrnNoiseP(iI), JstrnNoiseP(iI),fCoorSysNoiseP>0))./fCrct;
% end

