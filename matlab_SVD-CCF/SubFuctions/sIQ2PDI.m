%% Power Doppler processing
% input: sIQ, [nz,nx,nt]
% output: PDI, Power Doppler Image, [nz,nx,3],
    % 1: positive frequency, 
    % 2: negative frequency; 
    % 3: all frequency
function [PDI]=sIQ2PDI(sIQ)
[nz,nx,nt]=size(sIQ);
PDI=zeros(nz,nx,3); % 1: positive frequency; 2: negative frequency; 3: all frequency
nf= 2^nextpow2(2*nt);           % Fourier transform points
fIQ=fftshift(fft(sIQ,nf,3),3); % frequency of sIQHP

PDI(:,:,1)=(squeeze(mean(abs(fIQ(:,:,floor(nf/2)+1:nf)).^2,3))); % positive frequency, flowing down
PDI(:,:,2)=(squeeze(mean(abs(fIQ(:,:,1:floor(nf/2)-1)).^2,3)));  % negative frequency, flowing up
PDI(:,:,3)=(squeeze(mean(abs(sIQ(:,:,:)).^2,3)));                % all frequency

