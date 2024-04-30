%% IQ to sIQ with SVD data processign, sIQ to sIQHP with high pass filtering on sIQ.
% Input: 
    % IQ: complex IQ data, obtained with RF2IQ, [nz,nx,nt]
    % PRSSinfo.SVDrank: SVD rank [low high]
    % PRSSinfo.HPfC:  High pass filtering cutoff frequency, Hz
    % PRSSinfo.NEQ: do noise equalization? 0: no noise equalization; 1: apply noise equalization
    % PRSSinfo.rFrame: imaging frame rate, Hz
% output:
    % sIQ: SVD clutter rejected data, [nz,nx,nt]
    % sIQHP: SVD+HP clutter rejected data, [nz,nx,nt], cutoff frequency: PRSSinfo.HPfC
    % sIQHHP: SVD+HHP clutter rejected data, [nz,nx,nt], cutoff frequency: 70 Hz
% subfunction:
    % [sIQ, Noise]=SVDfilter(IQ,SignalRank)
function [sIQ, sIQHP, sIQHHP, eqNoise]=IQ2sIQ(IQ,PRSSinfo)
[nz, nx, nt]=size(IQ);
%% I. SVD filtering
[sIQ, Noise]=SVDfilter(IQ,PRSSinfo.SVDrank); % sIQ: signal I
eqNoise=medfilt2(abs(squeeze(mean(Noise,3))),[50 50],'symmetric');
eqNoise=eqNoise/min(eqNoise(:));
if PRSSinfo.NEQ==1
    sIQ=sIQ./repmat(eqNoise,[1,1,nt]); % gain compensation
end
%% SVD followed by HP
[B,A]=butter(4,PRSSinfo.HPfC/(PRSSinfo.rFrame/2),'high');    %coefficients for the high pass filter
sIQ1(:,:,101:100+nt)=sIQ;
sIQ1(:,:,1:100)=flip(sIQ1(:,:,101:200),3);
sIQ2=filter(B,A,sIQ1,[],3);    % blood signal (filtering in the time dimension)
sIQHP=sIQ2(:,:,101:end); % High pass filtered sIQ
clear sIQ1 sIQ2
%% SVD followed by higher cutoff frequency HP, for PDI calculation, large vessels
[B,A]=butter(4,70/(PRSSinfo.rFrame/2),'high');    %coefficients for the high pass filter
sIQ1(:,:,101:100+nt)=sIQ;
sIQ1(:,:,1:100)=flip(sIQ1(:,:,101:200),3);
sIQ2=filter(B,A,sIQ1,[],3);    % blood signal (filtering in the time dimension)
sIQHHP=sIQ2(:,:,101:end); % High pass filtered sIQ