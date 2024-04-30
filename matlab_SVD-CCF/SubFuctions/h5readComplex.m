%% h5 read function for complex data
% Note: use with the following code in front of the function
% dataName=h5read(saveName,'/dataName');
%% example
% saveName='D:\WeiYUN\Work\PROJ - OCT\CODE\OCT-OCTA\SubFunctions\test.h5'
% dataName=h5read(saveName,'/dataName');
% Data=h5readComplex(saveName,dataName);
%% 
function Data=h5readComplex(saveName,dataName)
% saveName=[filePath,'\',fileName,'.h5'];
cplxData=h5read(saveName, ['/Info-',dataName]);
if cplxData==1    
    realData=h5read([saveName],['/r',dataName]);
    imagData=h5read([saveName],['/i',dataName]);
    Data=single(realData)+1i*single(imagData);
else
    Data=single(h5read([saveName],['/',dataName]));
end

normValue=single(h5read([saveName],['/normValue_',dataName]));
Data=Data*normValue; % amplitude scale back
