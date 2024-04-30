%% convert vUS to PDI/SNR weighted hsv image
function PLOTwtV(V,PDI,Coor,VcAXIS)
% V(:,:,1:2): velocity, 1: up flow; 2: down flow
% PDI(:,:,1:3): PDI value, 1: up flow; 2: down flow; 3: all direction
% VcAXIS: [min, max], caxis range of V
%% %%%%%%%%%%% EXAMPLE %%%%%%%%%%%%%%%
% Coor.x=xCoor; Coor.z=zCoor;
% figure;
% PLOTwtV(V,PDI,Coor,[-30 30])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[VzCmap, VzCmapDn,VzCmapUp]=Colormaps_fUS;
if nargin<3
    VcAXIS=[-30 30];
end
Vup=-1*abs(V(:,:,1));
Vdn=1*abs(V(:,:,2));
PDIup=PDI(:,:,1);
PDIdn=PDI(:,:,2);
%% PLOT background (up flow, negative value) 
Vup(Vup<VcAXIS(1))=VcAXIS(1);
L = size(VzCmap,1);
% Scale the matrix to the range of the map.
Gup = round(interp1(linspace(VcAXIS(1),VcAXIS(2),L),1:L,Vup));
Hup = reshape(VzCmap(max(int16(Gup),1),:),[size(Gup) 3]); % Make RGB image from scaled.
% figure,image(H) % plot RGB figure
HSVup=rgb2hsv(Hup); % convert to HSV
% changes saturation
s=abs(Vup)/3;
s(s>1)=1;
% change value
% PDIup=PDIup./imgaussfilt(PDIup,20);
[nz, nx]=size(PDIup);
zPix=1:nz;
% SNR0=squeeze(PDIup);
% zSNR=(mean(SNR0,2)-0.3*std(SNR0,[],2));
% mskC=polyfit(zPix,zSNR',1);
% SNRthd=polyval(mskC,zPix);
% 
% for iz=1:nz
%     temp=PDIup(iz,:);
%     temp(temp<SNRthd(iz))=min(PDIup(:));
%     PDIup(iz,:)=temp;
% end
% %     PDIup(PDIup<mean(PDIup(:))+0.1*std(PDIup(:)))=min(PDIup(:));
vupMsk=(abs(Vup)/4).^2;
vupMsk(vupMsk>=1)=1;
% PDIup_thd=PDIup;
v=(PDIup-min(PDIup(:)))/(max(PDIup(:))-min(PDIup(:)));
v=v./(mean(v,2)-0.3*(1+(zPix./(4*nz)))'.*std(v,[],2));
v=(v/0.8).^2;
v(v>1)=1;
v=v.*vupMsk;
HSVup(:,:,2)=s;
HSVup(:,:,3)=v.^2;
image(Coor.x,Coor.z,hsv2rgb(HSVup))
axis equal
axis tight
xlabel('X [mm]')
ylabel('Z [mm]')
colorbar;
colormap(VzCmap)
caxis(VcAXIS)
%% PLOT overlap (down flow, positive value) 
Vdn(Vdn>VcAXIS(2))=VcAXIS(2);
L = size(VzCmap,1);
% Scale the matrix to the range of the map.
Gdn = round(interp1(linspace(VcAXIS(1),VcAXIS(2),L),1:L,Vdn));
Hdn = reshape(VzCmap(max(int16(Gdn),1),:),[size(Gdn) 3]); % Make RGB image from scaled.
% figure,image(H) % plot RGB figure
HSVdn=rgb2hsv(Hdn); % convert to HSV
% changes saturation
s=abs(Vdn)/5;
s(s>1)=1;
% change value
PDIdn=PDIdn./imgaussfilt(PDIdn,30);
v=(PDIdn-min(PDIdn(:)))/(max(PDIdn(:))-min(PDIdn(:)));
v=v./(mean(v,2)+0.6*(1-(zPix./(4*nz)))'.*std(v,[],2));
v(v>1)=1;
v(v<0.2)=0;
HSVdn(:,:,2)=s;
HSVdn(:,:,3)=v;
hAxes2 = axes;
%set visibility for axes to 'off' so it appears transparent
axis(hAxes2,'off')
h=image(Coor.x,Coor.z,hsv2rgb(HSVdn));
% set alpha
PDIMsk=zeros(size(PDIdn)); VMsk=PDIMsk;
PDIMsk(PDIdn>mean(PDIdn(:))-1*std(PDIdn(:)))=1;
VMsk=(Vdn/6).^2;
VMsk(VMsk>1)=1;
AlphaMsk=PDIMsk.*VMsk;
axis equal
axis tight
axis off
alpha(h,AlphaMsk)
colorbar;
colormap(VzCmap)
caxis(VcAXIS)