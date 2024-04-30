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
% h5writeComplex(saveName,char(dataName(1)),RR,1); % save RR to the .h5 file, the RR is complex data
% h5createComplex(saveName,char(dataName(2)),AG,'int16',0);% creat AG in the .h5 file, the AG is real data
% h5writeComplex(saveName,char(dataName(2)),AG,0); % save AG to the .h5 file, the AG is real data
%%
% dataSaveType: single, double, int16, uint16...
function h5createComplex(saveName,dataName,Data,dataSaveType,cplxData)
% saveName=[filePath,'\',fileName,'.h5'];
if nargin<4
    dataSaveType='single';
    cplxData=0;
end

if nargin<5
    cplxData=0;
end
Info.dataSize=size(Data);
Info.cplxData=cplxData;
Info.dataSaveType=dataSaveType;
h5create(saveName, ['/Info-',dataName],1,'Datatype','int8'); % save the datatype information, complex of real
if cplxData==1    
    h5create(saveName, ['/r',dataName],Info.dataSize ,'Datatype',dataSaveType);
    h5create(saveName, ['/i',dataName],Info.dataSize ,'Datatype',dataSaveType);    
else
    h5create(saveName, ['/',dataName],Info.dataSize ,'Datatype',dataSaveType);
end
h5create(saveName, ['/normValue_',dataName],1 ,'Datatype','single');
h5create(saveName, ['/reScalPrct_',dataName],1 ,'Datatype','single');
h5create(saveName, ['/dataType_',dataName],1 ,'Datatype','string');
h5write(saveName, ['/dataType_',dataName],string(dataSaveType));
% save([saveName(1:end-3), '-Info.mat'],'Info');
