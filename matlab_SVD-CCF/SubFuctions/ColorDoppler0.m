%% color Doppler data processing to get axial blood flow velocity
% input: 
    % sIQ: bulk motion removed data
    % PRSSinfo: data acquistion information, including
        % PRSSinfo.rFrame: sIQ frame rate, Hz
        % PRSSinfo.f0: Transducer center frequency, Hz
        % PRSSinfo.C: Sound speed in the sample, m/s
% output:
    % Vcz: axial velocity calculated with Color Dopler, mm/s 
function [Vcz]=ColorDoppler(sIQ,PRSSinfo)
[np,nt]=size(sIQ);
fCoor=linspace(-PRSSinfo.rFrame/2,PRSSinfo.rFrame/2,nt);

fBlood=fftshift(fft(sIQ,nt,2),2);
% fBlood(abs(fBlood)<0.8e8)=0; % thresholding
fD=sum(repmat(permute(fCoor,[1,2]),[np,1]).*abs(fBlood).^2,2)./sum(abs(fBlood).^2,2);
Vcz=fD.*PRSSinfo.C/(2*PRSSinfo.f0)*1e3; % axial speed obtained with color Doppler, mm/s