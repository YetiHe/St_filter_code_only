% sIQ2PDI
function [PDI]=sIQ2PDI_GPU(sIQ)
[nz,nx,nt]=size(sIQ);
sIQ=gpuArray(sIQ);
PDI=zeros(nz,nx,3,'gpuArray'); % 1: positive frequency; 2: negative frequency; 3: all frequency
nf= 2^nextpow2(2*nt);           % Fourier transform points
fIQ=fftshift(fft(sIQ,nf,3),3); % frequency of sIQHP

PDI(:,:,1)=(squeeze(mean(abs(fIQ(:,:,floor(nf/2)+1:nf)).^2,3))); % positive frequency, flowing down
PDI(:,:,2)=(squeeze(mean(abs(fIQ(:,:,1:floor(nf/2)-1)).^2,3)));  % negative frequency, flowing up
PDI(:,:,3)=(squeeze(mean(abs(sIQ(:,:,:)).^2,3)));                % all frequency
PDI=gather(PDI);

