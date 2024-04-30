%% singular value decomposition filter function
% input:
    % IQ data, [nz,nx,nt]
    % SignalRank, SVD rank, [Low High]
% output:
    % sIQ: SVD clutter rejected signal, [nz,nx,nt]
    % Noise: SVD-based low spatial-temporal correlation noise data
function [sIQ, Noise]=SVDfilter(IQ,SignalRank)
[nz,nx,nt]=size(IQ);
S=reshape(single(IQ),[nz*nx,nt]);
S_COVt=(S'*S);
[V,D]=eig(S_COVt); % V is the right singular Vector of S/eigenvector; D is the eigenvalue/square of Singular value
Ddiag=diag(abs(sqrt(D)));
Ddiag=20*log10(Ddiag/max(Ddiag)); % singular value in db
[Ddesc, Idesc]=sort(Ddiag,'descend');
Vdesc=V(:,Idesc);
UDelta=S*Vdesc;
%% SVD filtered
Vrank=zeros(size(Vdesc));
rank=SignalRank(1):SignalRank(2);
Vrank(:,rank)=Vdesc(:,rank);
sIQ=reshape(UDelta*Vrank',[nz,nx,nt]);
%% Noise 
Vnoise=zeros(size(Vdesc));
Vnoise(:,end-50:end)=Vdesc(:,end-50:end);
Noise=reshape(UDelta*Vnoise',[nz,nx,nt]);
sNoiseMed=medfilt2(abs(squeeze(mean(Noise,3))),[50 50],'symmetric');
Noise=sNoiseMed/min(sNoiseMed(:));

