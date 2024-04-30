%% autocorrelation calculation, CPU
% DAT: 2D matrix
% dim: specify which dimension to perform autocorrelation 
function ACF = aCorr(DAT, nTau, dim)
if dim==2
    DAT=DAT.';
end
[nt, nD] = size(DAT) ;
DATcj=conj(DAT(:,:));
Deno=sum((DATcj.*(DAT(:,:))),1);
for itau = 1:nTau
    Numer(itau,:)=sum((DATcj(1:nt-itau,:)).*(DAT(itau:nt-1,:)),1);
end
ACF = Numer./Deno;
