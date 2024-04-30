%% singular value decomposition filter function
function [SignalData, Noise]=SVDfilter_GPU(Data,SignalRank)
[nz,nx,nt]=size(Data);
S=reshape(gpuArray(single(Data)),[nz*nx,nt]);
S_COVt=(S'*S);
[V,D]=eig(S_COVt); % V is the right singular Vector of S/eigenvector; D is the eigenvalue/square of Singular value
Ddiag=diag(abs(sqrt(D)));
Ddiag=20*log10(Ddiag/max(Ddiag)); % singular value in db
[Ddesc, Idesc]=sort(Ddiag,'descend');
Vdesc=V(:,Idesc);
UDelta=S*Vdesc;
%% SVD filtered
Vrank=gpuArray(zeros(size(Vdesc)));
rank=SignalRank(1):SignalRank(2);
Vrank(:,rank)=Vdesc(:,rank);
SignalData=gather(reshape(UDelta*Vrank',[nz,nx,nt]));
%% Noise 
Vnoise=gpuArray(zeros(size(Vdesc)));
Vnoise(:,end-50:end)=Vdesc(:,end-50:end);
sNoise=gather(reshape(UDelta*Vnoise',[nz,nx,nt]));
sNoiseMed=medfilt2(abs(squeeze(mean(sNoise,3))),[50 50],'symmetric');
Noise=sNoiseMed/min(sNoiseMed(:));

