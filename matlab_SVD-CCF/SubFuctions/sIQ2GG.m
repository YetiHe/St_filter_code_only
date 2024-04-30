%% g1 calculation
% input:
    % sIQ: bulk motion removed data, [nz,nx,nt]
    % PRSSinfo: data processing parameters, including 
        % PRSSinfo.g1StartT: g1 calculation start number
        % RPSSinfo.g1nT: g1 calculation sample number
        % PRSSinfo.g1nTau: maximum number of time lag
% Output:
    % GG: g1 of sIQ, [nz,nx,nTau]
% Jianbo Tang, 20190819
function GG = sIQ2GG(sIQ, PRSSinfo)

%% constant total number of samples, nt
[nz, nx, nxRpt] = size(sIQ) ;
if PRSSinfo.g1nT>nxRpt-PRSSinfo.g1nTau
    PRSSinfo.g1nT=nxRpt-PRSSinfo.g1nTau-PRSSinfo.g1StartT+1;
%     disp(['Warning: nt is larger than nxRpt-ntau, and is modified to be nxRpt-ntau=',num2str(PRSSinfo.g1nT),'!']);
end
%% calculate g1
%%%% g1 = mean((yi*)*(y(i+tau))/mean(yi**yi);
temp_deno=(conj(sIQ(:,:,PRSSinfo.g1StartT:PRSSinfo.g1StartT-1+PRSSinfo.g1nT))).*(sIQ(:,:,PRSSinfo.g1StartT:PRSSinfo.g1StartT-1+PRSSinfo.g1nT));
for itau = 1:PRSSinfo.g1nTau
    Numer(:,:,itau)=mean((conj(sIQ(:,:,PRSSinfo.g1StartT:PRSSinfo.g1StartT-1+PRSSinfo.g1nT))).*(sIQ(:,:,itau+PRSSinfo.g1StartT:itau+PRSSinfo.g1StartT-1+PRSSinfo.g1nT)),3);
    itau+PRSSinfo.g1StartT-1+PRSSinfo.g1nT;
end
% size(Numer)
Denom=repmat(mean(temp_deno,3),[1,1,PRSSinfo.g1nTau]); % calculate the denomenator
GG = Numer./Denom;

