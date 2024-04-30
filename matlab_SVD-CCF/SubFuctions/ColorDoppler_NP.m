%% color Doppler data processing to get axial blood flow velocity
% input: 
    % sIQ: bulk motion removed data
    % PRSSinfo: data acquistion information, including
        % PRSSinfo.rFrame: sIQ frame rate, Hz
        % PRSSinfo.f0: Transducer center frequency, Hz
        % PRSSinfo.C: Sound speed in the sample, m/s
% output:
    % Vcz: axial velocity calculated with Color Dopler, mm/s 
function [Vcz]=ColorDoppler_NP(sIQ,PRSSinfo)
[nz,nx,nt]=size(sIQ);
fCoor=linspace(-PRSSinfo.rFrame/2,PRSSinfo.rFrame/2,nt);

fBlood=fftshift(fft(sIQ,nt,3),3);
% fBlood(abs(fBlood)<4*std(abs(fBlood(abs(fBlood)>0))))=0; % thresholding
Dirc=sign(sum(squeeze(mean(mean(abs(fBlood),1),2))'.*fCoor));
HfreqNoise=abs(fBlood(:,:,Dirc*(fCoor)>1500));

% fBlood(abs(fBlood)<(mean(HfreqNoise,3)+5*std(HfreqNoise,[],3)))=0; % thresholding
fBlood(abs(fBlood)<(max(abs(HfreqNoise),[],3)*1.3))=0; % thresholding
% fBlood(abs(fBlood)<4*mean(abs(fBlood(abs(fCoor)>1000))))=0; % thresholding
fD=sum(repmat(permute(fCoor,[1,3,2]),[nz,nx,1]).*abs(fBlood).^2,3)./sum(abs(fBlood).^2,3);
Vcz=fD.*PRSSinfo.C/(2*PRSSinfo.f0); % axial speed obtained with color Doppler, mm/s
Vcz(isnan(abs(Vcz)))=0;