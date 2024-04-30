%% 生成并保存CC数据：IQ图+对应的分割掩码（1号label）+对应的滤波后图（2号label或完整label）；不需要RF图
% clear;
DefPath='I:\ultrasound1\Yongchao\Data\Phantom data\hyt20240111_phamton';
codePath=pwd;
addpath([codePath,'\SubFuctions']);
parallel.gpu.enableCUDAForwardCompatibility(true)
datasetpath=uigetdir(DefPath, "select the CC data document path");
setInfo=strsplit(datasetpath(1:end),'\');
set_name = setInfo{end};
datapath=uigetdir(DefPath, "select the raw data document path");
rawInfo=strsplit(datapath(1:end),'\');
raw_name = rawInfo{end};

% 创建新的子数据集
subdataset_dir = datasetpath+"\"+raw_name;
if exist(subdataset_dir) == 0
    mkdir(subdataset_dir)
end

RFpath=uigetfile(datapath, "select the first rf file");
RFInfo=strsplit(RFpath(1:end),'_');
RF_name = RFInfo{end};

IQpath=uigetfile(datapath, "select the first IQ file");
IQInfo=strsplit(IQpath(1:end),'-');
IQ_name = IQInfo{end};

Infoname=uigetfile(datapath, "select the P file");
load(datapath+"\"+Infoname);  % 名字叫做P
%%
prompt={'number of frames to share the same CC threshold (be able to divide 1000)',...
    'files begin','files number','frames to pile (to resist to the noise)','do u want to segment the vessel wall manually?(y/n)'};
name='hyt data processing';
defaultvalue={'500','1','54','1','y'};
numinput=inputdlg(prompt,name, 1, defaultvalue);
frames_group_size = str2num(numinput{1});
file_begin = str2num(numinput{2});
file_num = str2num(numinput{3});
noise_stack = str2num(numinput{4});
wall_mode = numinput{5};
if mod(1000,frames_group_size) ~= 0
    frames_group_size = 100;
end

params.file_num = file_num;
params.frames_group_size = frames_group_size;
params.noise_stack = noise_stack;
params.file_begin = file_begin;
if wall_mode == 'n'
    params.label_mode = "automatic";
    prompt={'preset vessel width (mm; for automatic mode)'};
name='hyt data processing';
defaultvalue={'5'};
numinput=inputdlg(prompt,name, 1, defaultvalue);
preset_width_half = str2num(numinput{1})/2;
params.preset_width_half = preset_width_half ;
else
    params.label_mode = "manual";
end

%%
if params.label_mode == "manual"
    % 生成视频与mlp并要求手动画出掩码区域
    IQ_previous = "";
    % for i = 1:8
    %    IQ_previous = IQ_previous + IQInfo{i} + "-"; 
    % end
    % IQ = struct("data",[]);
    % for ifile = 1:params.file_num
    % % ifile = 1;
    %     IQgroup_temp=h5readComplex([datapath+"\"+IQ_previous+num2str(ifile)+"-"+IQInfo{10}],['IQ']);
    %     IQ(ifile).data = mean(IQgroup_temp,3);
    % end
    % disp("video is beginning in 3 seconds, and pls pay attention to the vessel wall's places...")
    % pause(3);
    % figure(33)
    % for i = 1:length(IQ)
    %     frame = IQ(i).data;
    %     figure(1000);
    %     imagesc(abs(squeeze(log10(frame))))
    %     %caxis([2.3 4])
    %     colormap bone
    %     title(i/18)
    %     pause(10/18)
    % end
    
    tmp_RF_name = RFInfo{1}+"_Frame"+num2str(1)+"_"+RFInfo{3};
    RF_sample=max(abs(h5readComplex([datapath+"\"+tmp_RF_name],['EG1'])),[],3);
    manual_mask = zeros([size(RF_sample,1),size(RF_sample,2)]);
    
    mask_operation = 1;

    while mask_operation ~= 0
    % figure(1000);
    figure(1001);
    imagesc(RF_sample);
    colormap gray;
    colorbar;
    if mask_operation == 1
        title("pls draw the contours which is to add to the original mask (focused place is recommended)");
    elseif mask_operation == 2
        title("pls draw the contours which is to be subtracted by the original mask (focused place is recommended)");
    end
    h = drawfreehand;
    if mask_operation == 1 || mask_operation == 3
        manual_mask = manual_mask + h.createMask();
    elseif mask_operation == 2
        manual_mask = manual_mask - h.createMask();
    end
    manual_mask(manual_mask >= 0.5) = 1;
    manual_mask(manual_mask < 0.5) = 0;
    figure(1002);
    imagesc(manual_mask);
    title("the mask u draw.");
    prompt={'choose the operation for the mask: 0 = finish; 1 = add another mask; 2 = reduce another mask; 3 = redraw the all masks'};
    name='hyt data processing';
    defaultvalue={'0'};
    numinput=inputdlg(prompt,name, 1, defaultvalue);
    mask_operation = str2num(numinput{1});
    if mask_operation == 3
        manual_mask = zeros([size(RF_sample,1),size(RF_sample,2)]);
    end

    end
    % 画的mask最好只包含有焦点的那一半（比如焦点找到上半wall就只画上半wall，下半wall不能画，因为CC算法如果信噪比低会带来噪音）

    params.manual_mask = manual_mask;
end

%% 基本信息输入脚本
skip = frames_group_size;
total_frame = file_num*1000/skip;
maps = struct("map",[],"v_mid",[],"v_max",[],"v_fast_mid",[]);
maps = repmat(maps,[1,total_frame]);
maps(1) = [];

coarse_A_window_number = 16; % [x,y], i.e.the number of the window along the both directions
accuracy_A_window_number = 4; % [x,y], i.e.the number of the window along both directions
accuracy_step_occupy = 0.75; % overlap ratio
sampling_fre = P.CCFR/skip;
params.CCFR = P.CCFR;
params.coarse_A_window_number = coarse_A_window_number;
params.accuracy_A_window_number = accuracy_A_window_number;
params.accuracy_step_occupy = accuracy_step_occupy;
% 校准两图轴向位置
% z_offset = 40;

dz = P.vSound/double(P.sampling*1e6); % m = t*v, RF的dz，不是IQ的dz
dx = P.pitch * 1e-3; % m，RF的dx，不是IQ的dx
resolution_RF = [dx, dz]*1e3; % mm\

% dz_IQ = P.zCoor(2) - P.zCoor(1); % IQ的dz
% dx_IQ = P.xCoor(2) - P.xCoor(1); % IQ的dx
% resolution_IQ = [dx_IQ, dz_IQ]; % mm, [dx,dz]

lag = 0;
dt_RF=1/double(P.sampling*1e6);                                 % The interval time between adjacent sampling points
rf_time_vector=P.Toffset+(1:1:P.actZsamples)*dt_RF-lag*dt_RF; 
iq_decimation_factor = 1;%round(double(P.sampling)/P.frequency);
demodulation_frequency =P.frequency*1e6;

%% 循环 1-54
% 新发现！散斑本身会影响CC算法结果，建议先用SVD生成散斑血流掩码掩盖掉血管内的位移
% 为了内存考虑，一组一组处理
for group_id = file_begin:file_begin+file_num-1
% group_id = 1;  % 改for

RF_group_name = "Frame"+num2str(group_id);
IQ_group_name = num2str(group_id);
tmp_RF_name = RFInfo{1}+"_"+RF_group_name+"_"+RFInfo{3};
% tmp_IQ_name = "";
% for i = 1:8
%    tmp_IQ_name = tmp_IQ_name + IQInfo{i} + "-"; 
% end
% tmp_IQ_name = tmp_IQ_name + IQ_group_name + "-";
% tmp_IQ_name = tmp_IQ_name + IQInfo{10};

RFgroup_temp=h5readComplex([datapath+"\"+tmp_RF_name],['EG1']);
%IQgroup_temp=h5readComplex([datapath+"\"+tmp_IQ_name],['IQ']);


for img_id = 1:total_frame/file_num  % 改for

% 从group中读取
% current_frame_ori =mean(double(abs(IQgroup_temp(:,:,(img_id-1)*skip+1:(img_id-1)*skip+noise_stack))),3);%abs(mean(IQ1(:,:,1:round(size(IQ1,3)/2)),3)); %abs(IQ1(:,:,1));
% next_frame_ori = mean(double(abs(IQgroup_temp(:,:,img_id*skip-noise_stack+1:img_id*skip))),3);%abs(mean(IQ1(:,:,round(size(IQ1,3)/2)+1:end),3));%abs(IQ2(:,:,2));
current_RF_frame = mean(double(RFgroup_temp(:,:,(img_id-1)*skip+1:(img_id-1)*skip+noise_stack)),3);%mean(RF1(:,:,1:round(size(IQ1,3)/2)),3);%RF1(:,:,1);
next_RF_frame = mean(double(RFgroup_temp(:,:,img_id*skip-noise_stack+1:img_id*skip)),3);%mean(RF1(:,:,round(size(IQ1,3)/2)+1:end),3);%RF2(:,:,2);
current_frame = abs(demodulate(rf_time_vector',current_RF_frame,demodulation_frequency,1));
next_frame = abs(demodulate(rf_time_vector',next_RF_frame,demodulation_frequency,1));

% current_frame = imresize(current_frame_ori,size(current_RF_frame)); % 调为一致
% next_frame = imresize(next_frame_ori,size(next_RF_frame));

% current_frame = current_frame(1:end-z_offset,:).*mask;
% current_RF_frame = current_RF_frame(1+z_offset:end,:).*mask;
% next_frame = next_frame(1:end-z_offset,:).*mask;
% next_RF_frame = next_RF_frame(1+z_offset:end,:).*mask;

    % 制作基于高rank的svd滤波的PDI掩码来遮蔽掉高速血流导致的位移，因为我们只想统计血管壁的位移情况
% sIQ = double(abs(RFgroup_temp(:,:,(img_id-1)*skip+1:img_id*skip)));
% [filter_sIQ,noise] = SVDfilter(sIQ,[svd_rank,size(sIQ,3)]); %[min(round(size(sIQ,3)*svd_rank_ratio),25),size(sIQ,3)]
% PDI = sIQ2PDI(filter_sIQ);
% tmp = squeeze(PDI(:,:,3));
% mask_ori = tmp;
% thre_tmp = mean(tmp(:))+2*std(tmp(:));
% mask_ori(tmp>=thre_tmp) = 1; %血流部分的位移，后期需要反色遮蔽
% mask_ori(tmp<thre_tmp) = 0;
% se = strel('disk', 5); % 形态学滤波
% mask_ori = imopen(mask_ori,se);
% mask_ori = imopen(mask_ori,se);
% mask_ori = imclose(mask_ori,se);
% mask_ori = imclose(mask_ori,se);
% mask_ori = imfill(mask_ori); % 填充空洞
% mask = -mask_ori + 1; % 反色
% figure(3);
% imagesc(mask);
% 注意，仅仅只是屏蔽算出来的结果可能没有用，因为final——map已经是受到血流散斑影响后计算出来的伪结果了，
% 如果试出来没用的话，看之前那篇论文怎么处理（好像有手动掩码），或在CC的输入之前就掩码成0
% 如果基于PDI的掩码作用不大的话，找论文看看如何识别血管壁（轴向即可），只对血管壁掩码为1也可以

% coarse_resolution_scale_factor = (size(current_frame)./size(current_frame_ori));
coarse_resolution = resolution_RF; %./ [coarse_resolution_scale_factor(2),coarse_resolution_scale_factor(1)]; % unit: mm, [dx,dz]
accurate_resolution = resolution_RF(2); % unit: mm
params.coarse_resolution = coarse_resolution;
params.accurate_axial_resolution = accurate_resolution;
% figure(3);
% imagesc(current_frame);
% colormap turbo
% figure(4);
% imagesc(current_RF_frame);
% colormap turbo

final_map  = hyt_CC_V2(current_frame,next_frame,current_RF_frame,next_RF_frame,...
    coarse_A_window_number,accuracy_A_window_number,accuracy_step_occupy,coarse_resolution,...
    accurate_resolution,[]);

maps((group_id-1)*total_frame/file_num+img_id - (file_begin-1)*1000/skip).map = final_map;
disp(strcat(num2str((group_id-1)*total_frame/file_num+img_id - (file_begin-1)*1000/skip),"/",num2str(total_frame)));

end

end
save("temp.mat","maps");
save(subdataset_dir+"\"+raw_name+"_CC_data.mat","maps","params");

%% 离线处理final map, 准备加入血管壁注意力机制
load("temp.mat");
%load("I:\ultrasound1\Yongchao\Data\Phantom data\hyt20240111_phamton\hyt_dataset\V0-A0-PRF18K\V0-A0-PRF18K_CC_data-skip500-axialsearch0301.mat");
for i = 1:length(maps)
    final_map  = maps(i).map; % 单位：mm

cal_temp = real(final_map);
[mid_dx,max_dx,fast_mid_dx,slow_mid_dx] = hyt_statistc_for_CC(cal_temp,[]);

if params.label_mode == "automatic"
    group_id = ceil(i/(1000/skip));
    RF_group_name = "Frame"+num2str(group_id);
    tmp_RF_name = RFInfo{1}+"_"+RF_group_name+"_"+RFInfo{3};
    RFgroup_temp=h5readComplex([datapath+"\"+tmp_RF_name],['EG1']);
    img_id = i - (group_id-1)*(1000/skip); % 复原
    RF_tmp = mean(double(RFgroup_temp(:,:,(img_id-1)*skip+1:img_id*skip)),3);
    IQ_tmp = abs(demodulate(rf_time_vector',RF_tmp,demodulation_frequency,1));

    preset_half = preset_width_half; %mm, 1.5mm适配人类动脉，仿体的胶管很粗，输入2.5mm（大约5mm粗的管壁）
    pixel_window_half = round(preset_half/resolution_RF(2)); % 前后n个pixel来搜索
    [~,axial_gradient] = gradient(IQ_tmp);

    moving_win_len = pixel_window_half; %pixel
    axial_gradient_2order = zeros(size(axial_gradient));
    for win_id = 1:moving_win_len:size(axial_gradient,1)
        win_segment = axial_gradient(win_id:min(win_id+moving_win_len-1, size(axial_gradient,1)),:);
        max_value = max(win_segment,[],1);
        axial_gradient_2order(win_id:min(win_id+moving_win_len-1, size(axial_gradient,1)),:) = repmat(max_value,size(win_segment,1),1);
    end

    b_map = zeros(size(IQ_tmp));
    se = strel('line',pixel_window_half*2,90);
    for ich = 1:size(IQ_tmp,2)
        tmp1 = (axial_gradient_2order(:,ich)); % abs?
        global_thre = mean(tmp1) + 1*std(tmp1);

        b_map_tmp = tmp1;
        b_map_tmp(tmp1<global_thre) = 0;
        b_map_tmp(tmp1>=global_thre) = 1;
        b_map_tmp = imclose(b_map_tmp,se);
        b_map(:,ich) = b_map_tmp;
    end
    se2 = strel('cube',round(pixel_window_half/2));
    b_map = imclose(b_map,se2);
    b_map = imopen(b_map,se2);

    cal_temp = imag(final_map);
    [mid_dz,max_dz,fast_mid_dz] = hyt_statistc_for_CC(cal_temp,b_map);
else
    cal_temp = imag(final_map);
    [mid_dz,max_dz,fast_mid_dz,slow_mid_dz] = hyt_statistc_for_CC(cal_temp,params.manual_mask);
end

fast_mid_v = (fast_mid_dx + 1i*fast_mid_dz)*sampling_fre*1e-3; % m/s; 非常重要
slow_mid_v = (slow_mid_dx + 1i*slow_mid_dz)*sampling_fre*1e-3; % m/s; 非常重要
max_v = (max_dx+1i*max_dz)*sampling_fre*1e-3; % m/s; 非常重要
mid_v = (mid_dx+1i*mid_dz)*sampling_fre*1e-3; % m/s; 非常重要

maps(i).map = final_map;
maps(i).v_mid = mid_v;
maps(i).v_fast_mid = fast_mid_v;
maps(i).v_slow_mid = slow_mid_v;
maps(i).v_max = max_v;
disp(strcat("offline data processing: ",num2str(i),"/",num2str(length(maps))));
end
save("temp.mat","maps");
save(subdataset_dir+"\"+raw_name+"_CC_data.mat","maps","params");

%% 离线画图
load("temp.mat");
%
v_max_all = [maps(:).v_max]; % m/s
v_mid_all = [maps(:).v_mid];
v_fast_mid_all = [maps(:).v_fast_mid];
v_slow_mid_all = [maps(:).v_slow_mid];

v_max = imag(v_max_all); % 最大值受到降采样影响很大
v_mid = imag(v_mid_all); % 横向位移估计方案受限于横向分辨率，并不靠谱，先只看轴向位移估计
v_fast_mid = imag(v_fast_mid_all);
v_slow_mid = imag(v_slow_mid_all);

v_max_mean = mean(v_max(abs(v_max)>0))*1e2; % cm/s
if isnan(v_max_mean)
    v_max_mean = 0;
end
v_mid_mean = mean(v_mid(abs(v_mid)>0))*1e2;
if isnan(v_mid_mean)
    v_mid_mean = 0;
end

v_fast_mid_mean = mean(v_fast_mid(abs(v_fast_mid)>0))*1e2;
if isnan(v_fast_mid_mean)
    v_fast_mid_mean = 0;
end
v_slow_mid_mean = mean(v_slow_mid(abs(v_slow_mid)>0))*1e2;
if isnan(v_slow_mid_mean)
    v_slow_mid_mean = 0;
end

time_axis = 0:1/sampling_fre:length(maps)/sampling_fre-1/sampling_fre;
figure(33);
subplot(411);
p1 = plot(time_axis(2:end), v_mid(2:end)*1e2);
grid on
ylabel("V cm/s");
xlabel("time (s)");
title("axial median velocity magnitude (ignore direction).");
subplot(412);
p2 = plot(time_axis,v_max*1e2);
grid on
%legend([p1,p2],["bigger V's mean","only V max"]);
ylabel("V cm/s");
xlabel("time (s)");
title("velocity with axial maximum magnitude (ignore direction).");
subplot(413);
p3 = plot(time_axis,v_fast_mid*1e2);
grid on
%legend([p1,p2],["bigger V's mean","only V max"]);
ylabel("V cm/s");
xlabel("time (s)");
title("the faster half part of velocity's axial median magnitude (ignore direction).");
subplot(414);
p4 = plot(time_axis,v_slow_mid*1e2);
grid on
%legend([p1,p2],["bigger V's mean","only V max"]);
ylabel("V cm/s");
xlabel("time (s)");
title("the slower half part of velocity's axial median magnitude (ignore direction).");
saveas(gcf,subdataset_dir+"\"+"CC.jpg","jpg");
% 发现较慢的那一半比较容易藏有节拍，较快的那一半大概率是血流散斑带来的噪音
%% 离线放视频

% prompt={'do u want to see the movie explaining the displace changes accroding to your data? [y/n]'};
% name='hyt movie';
% defaultvalue={'n'};
% numinput=inputdlg(prompt,name, 1, defaultvalue);
% if numinput{1} == "y"
% 
% load("temp.mat");
% clim = 20;
% for i = 1:length(maps)
%     final_map  = maps(i).map; % 单位：mm
% 
% figure(100);
% 
% subplot(221);
% imagesc(imag(final_map));
% caxis([-clim,clim]);
% colorbar;
% %colormap('Colormaps_fUS');
% colormap turbo;
% 
% xlabel("CC axial (with direction, unit:mm)")
% subplot(222);
% imagesc(real(final_map));
% caxis([-clim,clim]);
% colorbar;
% xlabel("CC lateral (with direction, unit:mm)")
% 
% subplot(223)
% imagesc(abs(imag(final_map)));
% caxis([0,clim]);
% colorbar;
% xlabel("CC axial (without direction, unit:mm))")
% 
% subplot(224)
% imagesc(abs(real(final_map)));
% caxis([0,clim]);
% colorbar;
% xlabel("CC lateral (without direction, unit:mm))")
% 
% disp(strcat("frame id : ",num2str((i-1)*skip)))
% 
% pause(0.05)
% 
% end
% end
% 
% 








