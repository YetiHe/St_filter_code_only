function GG = sIQ2GG(sIQ, PRSSinfo)

%% constant total number of samples, nt
[nP, nxRpt] = size(sIQ);
if PRSSinfo.g1nT>nxRpt-PRSSinfo.g1nTau
    PRSSinfo.g1nT=nxRpt-PRSSinfo.g1nTau-PRSSinfo.g1StartT+1;
%     disp(['Warning: nt is larger than nxRpt-ntau, and is modified to be nxRpt-ntau=',num2str(PRSSinfo.g1nT),'!']);
end
%% calculate g1
%%%% g1 = mean((yi*)*(y(i+tau))/mean(yi**yi);
temp_deno=(conj(sIQ(:,PRSSinfo.g1StartT:PRSSinfo.g1StartT-1+PRSSinfo.g1nT))).*(sIQ(:,PRSSinfo.g1StartT:PRSSinfo.g1StartT-1+PRSSinfo.g1nT));
for itau = 1:PRSSinfo.g1nTau
    Numer(:,itau)=mean((conj(sIQ(:,PRSSinfo.g1StartT:PRSSinfo.g1StartT-1+PRSSinfo.g1nT))).*(sIQ(:,itau+PRSSinfo.g1StartT:itau+PRSSinfo.g1StartT-1+PRSSinfo.g1nT)),2);
    itau+PRSSinfo.g1StartT-1+PRSSinfo.g1nT;
end
% size(Numer)
Denom=repmat(mean(temp_deno,2),[1,PRSSinfo.g1nTau]); % calculate the denomenator
GG = Numer./Denom;

