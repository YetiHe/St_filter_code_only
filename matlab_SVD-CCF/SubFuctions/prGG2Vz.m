function [Vz]=prGG2Vz(GG, PRSinfo)
[nVox,nTau]=size(GG); % nVox=nz*nx*ny
%% OCT system parameter
dt=1/PRSSinfo.rFrame; % s
lambda=PRSSinfo.C/PRSSinfo.f0;        % wavlength, m
k0=2*pi/lambda; 
nz=PRSinfo.Dim(1);
nx=PRSinfo.Dim(2);
%% Vz calculation
GG0=GG-mean(real(GG(:,1:min(75,nTau))),2);
GGUnwrap=movmean(unwrap(angle(GG0),[],2),5,2);
maxPhs=abs((GGUnwrap(:,end)-GGUnwrap(:,1)));
ST=(GGUnwrap(:,end)-GGUnwrap(:,1))/nTau*1/2;      % Slope Threshold based on the difference of the last and fisrt unwraped phase
PS=diff(GGUnwrap,1,2);                            % Phase Difference of the slope
sRBC=bsxfun(@gt,PS.*sign(ST),ST.*sign(ST)*1.2);   % slope greater than the slope threshold, identified as RBC flowing through
aRBC=sum(PS.*sRBC,2);
% figure;imagesc(reshape(aRBC,[nz,nx]))
[ntRBC]=sum(sRBC,2);
% figure;imagesc(reshape(ntRBC,[nz,nx]))
w=aRBC./(ntRBC*dt);
% figure;imagesc(reshape(w,[nz,nx]))
Vz=w/(2*n*k0); % m/s
% figure;imagesc(Vz)
% colormap jet
% Vz=reshape(Vz,[nz*nx,1]); % m/s;
Vz(find(isnan(Vz)==1))=0;
% Vz=reshape(Vz,[nz,nx]); % m/s
% figure;imagesc(Vz)
% colormap jet
