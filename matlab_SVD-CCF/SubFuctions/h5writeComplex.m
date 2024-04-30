% saveName: The to be saved filePath with the fileName
% dataName: the variable/data name that to be saved in the saveName
% Data: the to be saved data
% cplxData: is the data to be saved as complex data (1) or real data (0)
% reScalPrct: [0 1], to normalize the range of the data; 
            %0: no normlization and no scaling
            %1: normalize to the maximum value of the data, and then scaled 
            %0.99: normalize to the 99% percntile value of the data, and then scaled (int16 range) 
%% h5 write function for complex data
% Note: use with the following code in front of the function
% h5create(iSaveName, '/dataName',length(dataName),'Datatype','string');  % save variable info in the .h5 file
% h5write(iSaveName,'/dataName',dataName);
%% example: save a complex data and a real data into a h5 file
% saveName='D:\WeiYUN\Work\PROJ - OCT\CODE\OCT-OCTA\SubFunctions\test.h5';
% RR=(single(rand(100,100)*100))+1i*(single(rand(100,100)*100)); % complex data
% AG=(single(rand(100,100)*100)); % real data
% dataName(1)=string('RR');
% dataName(2)=string('AG');
% h5create(saveName, '/dataName',length(dataName),'Datatype','string');  % save variable info in the .h5 file
% h5write(saveName,'/dataName',dataName);
% h5createComplex(saveName,char(dataName(1)),RR,'int16',1); % create RR in the .h5 file, the RR is complex data
% h5writeComplex(saveName,char(dataName(1)),RR,1,99); % save RR to the .h5 file, the RR is normalized to 99%, rescaled and then saved to complex data 
% h5createComplex(saveName,char(dataName(2)),AG,'int16',0);% creat AG in the .h5 file, the AG is real data
% h5writeComplex(saveName,char(dataName(2)),AG,0,0); % save AG to the .h5 file, the AG is not normalized, rescaled, and then saved to real data
%% 
function h5writeComplex(saveName,dataName,Data,cplxData,reScalPrct)
warning off
% saveName=[filePath,'\',fileName,'.h5'];
if nargin<4
    cplxData=0;
    reScalPrct=0; % for nomalization, 
end
if nargin<5
    reScalPrct=0; % for nomalization, 
end

dataSaveType=h5read(saveName,['/dataType_',dataName]);
if strcmp(dataSaveType,'int16')
    rScale=20000;
elseif strcmp(dataSaveType,'uint8')
    rScale=200;
elseif strcmp(dataSaveType,'int8')
    rScale=100;
else
    rScale=1;
end

h5write(saveName, ['/Info-',dataName],cplxData);
if reScalPrct==0
    normValue=max(abs(Data(:)))/rScale;
else  % normalization and scaling
    normValue=prctile(abs(Data(:)),reScalPrct*100)/rScale;    
end
Data=Data/normValue; % normalize to the percentile and rescale to the specified scale range of the dataType
h5write(saveName, ['/normValue_',dataName],single(normValue));

if cplxData==1
    realData=real(Data);
    imagData=imag(Data);
    h5write(saveName, ['/r',dataName],realData);
    h5write(saveName, ['/i',dataName],imagData);
else
    h5write(saveName, ['/',dataName],Data);
end
h5write(saveName, ['/reScalPrct_',dataName],reScalPrct);