%% 生成并保存数据集：IQ图+对应的分割掩码（1号label）+对应的滤波后图（2号label或完整label）；不需要RF图
clear;
DefPath='I:\ultrasound1\Yongchao\Data\Phantom data\hyt20240111_phamton';
codePath=pwd;
addpath([codePath,'\SubFuctions']);
parallel.gpu.enableCUDAForwardCompatibility(true)
[IQ_name, data_path]=uigetfile(DefPath, "select the first IQ file");
IQ_info = split(IQ_name,"-");
IQ_previous = "";
for i = 1:8
    IQ_previous = IQ_previous + IQ_info{i} +"-";
end

[CC_file, CC_path]=uigetfile(DefPath, "select the CC data mat");
load(CC_path+"\"+CC_file);  % 名字叫做P
CC_info = split(CC_path,"\");

subdataset_dir=uigetdir(DefPath, "select the dir of the sub dataset");
subdataset_path = subdataset_dir + "\" + CC_info{end-1} + "\train\";

if exist(subdataset_path,'dir') == 0
    mkdir(subdataset_path);
end

sampling_fre = params.CCFR/params.frames_group_size;
total_frames = 1000*params.file_num;
%%
prompt={'maximum rank','minimum rank','ratio between train:test'};
name='hyt data processing';
defaultvalue={'13','7','4'};
numinput=inputdlg(prompt,name, 1, defaultvalue);
high_rank = str2num(numinput{1});
low_rank = str2num(numinput{2});
ratio = str2num(numinput{3});
%%
img = imread(CC_path+"\CC.jpg");
figure(900);
imshow(img);
title("choose the channel")
prompt={'0 for axial v; 1 for faster half v; 2 for slower half v'};
name='hyt data processing';
defaultvalue={'2'};
numinput=inputdlg(prompt,name, 1, defaultvalue);
img_channel = str2num(numinput{1});
if img_channel == 0
    v_mid_all = [maps(:).v_mid];
elseif img_channel == 1
    v_mid_all = [maps(:).v_fast_mid];
else
    v_mid_all = [maps(:).v_slow_mid];
end
v_mid = imag(v_mid_all); % m/s
f_mid = fftshift(fft(v_mid-mean(v_mid)));
v_denoise = v_mid; % wdenoise(v_mid);
v_denoise(v_denoise < 0) = 0;
thre = prctile(v_denoise, [1,99]);
v_thre = v_denoise(v_denoise > thre(1) & v_denoise < thre(2)); % 去掉极值
svd_thre = mean(v_thre)+0.9*std(v_thre);

time_axis = 0:1/sampling_fre:length(maps)/sampling_fre-1/sampling_fre;
fre_axis = -sampling_fre/2:sampling_fre/length(f_mid):sampling_fre/2-sampling_fre/length(f_mid);
figure(1);
subplot(311)
plot(time_axis, v_mid);
hold on
grid on
plot(time_axis,repmat(svd_thre,[1,length(v_mid)]));
hold off
ylabel("V m/s");
xlabel("time (s)");
title("axial median velocity magnitude (ignore direction).");
subplot(312);
plot(fre_axis, abs(f_mid));
grid on
ylabel("abs frequency");
xlabel("frequency (Hz); subtracted DC value");
subplot(313);
plot(time_axis,v_denoise);
hold on
grid on
plot(time_axis,repmat(svd_thre,[1,length(v_denoise)]));
hold off
ylabel("V m/s");
xlabel("time (s)");
title("after wavelet denoise");
% 用去噪的v曲线配合v阈值来决定SVD阈值，虽然v阈值没变，但是小波去噪后的v曲线更平滑，更稳定。

%%

step = params.frames_group_size;
% 设定滤波器参数
cutoff_frequency = 30; % 截止频率为30Hz
order = 3; % 滤波器阶数
% 设计 Butterworth 高通滤波器
[b, a] = butter(order, cutoff_frequency/(params.CCFR/2), 'high');

for ifile = params.file_begin:params.file_begin+params.file_num-1
    disp(strcat("processing the ",num2str(ifile),"frame"));
  
    IQgroup_temp=h5readComplex([data_path+IQ_previous+num2str(ifile)+"-"+IQ_info{10}],['IQ']);
    tmp = sum(sum(abs(IQgroup_temp),2),3);
    zero_loc = find(tmp == 0); % 投影积分为0说明是图像重建成IQ时的衍生物，不是图像的一部分，应该切除,此衍生物一般成批出现在图像头或图像尾，确认该批IQ的衍生物集中在图像尾
    offset = length(zero_loc);
    IQcat = IQgroup_temp(1:end-offset,:,:);
    % 归一化
    % IQgroup = (IQcat - mean(IQcat(:)))/std(IQcat(:));
    IQgroup = IQcat / mean(abs(IQcat(:)));
    % frame1: mean:-8.2947 +16.6649i / 851.6365; std:1.3510e+03  / 1.0490e+03; range  -1.4512e+04 +1.2230e+04i
    % frame2: mean:  37.1325 +11.4999i /  853.9639; std: 1.3507e+03 /
    % 1.0472e+03; range  -1.5764e+04 +1.0332e+04i  /  1.8848e4
    %IQgroup = (IQcat - repmat(mean(mean(IQcat,1),2), [size(IQcat,1),size(IQcat,2),1]))./repmat(std(std(IQcat,[],1),[],2),[size(IQcat,1),size(IQcat,2),1]);
    for iframe = 0:step:1000-1
        sIQ1 = IQgroup(:,:,iframe+1:iframe+step); % 最后一个group了，直接一起上
        
        % 应用滤波器
        sIQ2 = permute(sIQ1,[3,1,2]); % t,z,x
        sIQ3 = filtfilt(b, a, sIQ2);
        sIQ = permute(sIQ3,[2,3,1]); % z,x,t

        step_index = 1000/step*(ifile-1)+1+iframe/step;
        v_factor = v_denoise(step_index);
        if v_factor <= svd_thre
            [filter_sIQ,~] = SVDfilter(sIQ,[round(low_rank+(v_factor-min(v_denoise))*(high_rank-low_rank)/(svd_thre-min(v_denoise))),size(sIQ,3)]);
            %[filter_sIQ,~] = SVDfilter(sIQ,[low_rank,size(sIQ,3)]);
        else
            [filter_sIQ,~] = SVDfilter(sIQ,[high_rank,size(sIQ,3)]);
        end
        PDI = sIQ2PDI(filter_sIQ);
        PDI_dB = 20*log10(PDI(:,:,3));
        mask = zeros(size(PDI_dB));
        mask(PDI_dB>=mean(PDI_dB(:))+2*std(PDI_dB(:))) = 1;
        % se = strel('disk', 5); % 形态学滤波
        % mask = imclose(mask,se);
        mask = imfill(mask);
        % mask = imopen(mask,se);

        label_group = filter_sIQ; %因为只有一个group，这个group内sIQ不是滑动的，所以取相应的帧就需要滑动
        sample_group = sIQ;
    
        for sub_id = 1:step
            frame_index = (step_index-1)*step + sub_id;
            label = label_group(:,:,sub_id);
            sample = sample_group(:,:,sub_id);
            save(strcat(subdataset_path,num2str(frame_index),".mat"),"label","sample","mask");
        end
    end
end

%% 分割数据集
test_id_begin = params.file_num * 1000 *ratio/(ratio+1) + params.file_begin;
% subdataset_dir_info = split(subdataset_dir,"\");
% test_sub_path = "";
% for i = 1:length(subdataset_dir_info)-1
%     test_sub_path = test_sub_path + subdataset_dir_info{i} + "\";
% end
test_sub_path = subdataset_dir + "\" + CC_info{end-1} + "\test";%CC_info{end-1}+ "\test";
if exist(test_sub_path,'dir') == 0
    mkdir(test_sub_path);
end

for test_id = test_id_begin:(params.file_begin+params.file_num-1)*1000
    file_path_name = strcat(subdataset_path,num2str(test_id),".mat");
    movefile(file_path_name, test_sub_path);
end

    
%% 查看视频
datapath=uigetdir(DefPath, "select the raw data document path");
rawInfo=strsplit(datapath(1:end),'\');
raw_name = rawInfo{end};

IQ = struct("data",[],"mask",[]);
IQ(1) = [];
tmp = zeros([size(label,1), size(label,2), 500]);
tmp2 = zeros([size(label,1), size(label,2), 500]);
cnt = 1;
for ifile = 1:54000
% ifile = 1;
    load(datapath+"\"+num2str(ifile)+".mat", "label","mask");
    tmp2(:,:,cnt) = mask;
    tmp(:,:,cnt) = 20*log10(abs(label).^2);
    if cnt == 500
        IQ(end+1).data = mean(tmp,3);

        IQ(end).mask = max(tmp2,[],3);
        tmp = zeros([size(label,1), size(label,2), 500]);
        tmp2 = zeros([size(label,1), size(label,2), 500]);
        cnt = 1;
    end
    cnt = cnt + 1;
end
disp("video is beginning in 3 seconds, and pls pay attention to the vessel wall's places...")
pause(3);

myVideo=VideoWriter(datapath+"\"+"sIQ"); % 指定视频文件名
myVideo.FrameRate = 18000/500;
open(myVideo); % open开始写入

for i = 1:length(IQ)
    frame = IQ(i).data;
    mk = IQ(i).mask;
    figure(1000);
    subplot(211)
    imagesc(frame)
    %caxis([2.3 4])
    colormap turbo
    colorbar
    axis equal
    title(strcat("time: ",num2str(i/36), " s"))
    subplot(212);
    imshow(mk);
    axis equal

    % pause(10/18)
    Fr=getframe(gcf); % 抓取图窗
    writeVideo(myVideo,Fr); % 写入文件
end

close(myVideo);
%% 用1000作为单次SVD的输入序列数，但是只取前n帧作为对应帧的标签，n=此前定义的降采样倍数，然后step=n
sIQ1 = IQgroup(:,:,1:500);
%sIQ =  IQgroup(:,:,1:500) - sIQ1;

% 设定滤波器参数
cutoff_frequency = 30; % 截止频率为30Hz
order = 3; % 滤波器阶数

% 设计 Butterworth 高通滤波器
[b, a] = butter(order, cutoff_frequency/(params.CCFR/2), 'high');

% 应用滤波器
sIQ2 = permute(sIQ1,[3,1,2]); % t,z,x

sIQ3 = filtfilt(b, a, sIQ2);

sIQ = permute(sIQ3,[2,3,1]); % z,x,t

figure(345);
imagesc(abs(sIQ1(:,:,1)));
colorbar;
colormap turbo;
title("before filtering")
figure(346);
imagesc(abs(sIQ(:,:,1)));
colorbar;
colormap turbo;
title("after filtering")
%%


[filter_sIQ,noise] = SVDfilter(sIQ,[6,size(sIQ,3)]);

%[filter_sIQ,noise] = SVDfilter(sIQ,[22,size(sIQ,3)]);
% round(low_rank+(v_factor-min(v_denoise))*(high_rank-low_rank)/(svd_thre-min(v_denoise)))
% aa = reshape(sIQ, [130*385,500]);
% figure(33);imagesc(abs(aa));colormap turbo; colorbar
PDI = sIQ2PDI(filter_sIQ);
PDI_all = PDI(:,:,3);
mask = zeros(size(PDI_all));
fsIQ = mean(abs(filter_sIQ),3);
PDI_dB = 20*log10(PDI_all);
mask(PDI_dB>=mean(PDI_dB(:))+2*std(PDI_dB(:))) = 1;
mask = imfill(mask);
% mask = imopen(mask,se);
% h=imrect;
% Position=round(getPosition(h));
% 第二个人，上限20
% svd按照500来算: 高速区：下限7-8(基本确定), 上限20; 设置活体上限10
% 低速区：上限20；下限5（基本确定）; 设置活体下限6
% 活体数据另算
% p1（左右）：上限12， 下限7
% p2: 上限17， 下限11(left，建议丢掉)
% p2（right）: 人动了，svd没法提取血流信号了，脏数据，剔除
% p3(left): 上限13，下限8; (人也动了，只是动的比较慢，如果数据量够用，建议剔除该数据)
% p3(right)：上限11，下限6
% p4（left）：数据质量差
% p4 (right): 上限13，下限7
figure(99);
subplot(411);
imagesc(mean(abs(sIQ),3));
axis equal
colorbar;
colormap turbo;
xlabel("IQ")
subplot(412);
imagesc(PDI_dB);
axis equal
colorbar;
colormap turbo;
xlabel("PDI in dB")
subplot(413);
imagesc(fsIQ);
axis equal
colorbar;
colormap turbo;
xlabel("averaged sIQ's magnitude")
subplot(414);
imshow(mask)
axis equal
xlabel("mask based on the PDI in dB")

sIQ2 = sIQ2PDI(sIQ);
sIQ2 = sIQ2(:,:,3);
sIQ2 = sIQ2.*mask;
% sIQ2 = (sIQ2 - min(sIQ2(:)))/(max(sIQ2(:)) - min(sIQ2(:)));
figure(3322);
subplot(211)
imagesc(mean(abs(sIQ),3));
colorbar;
colormap turbo;
subplot(212)
imagesc(20*log10(sIQ2));
colorbar;
colormap turbo;
%%
% sIQ = filter_sIQ;
% [nz,nx,nt]=size(sIQ);
% PDI=zeros(nz,nx,3); % 1: positive frequency; 2: negative frequency; 3: all frequency
% nf= 2^nextpow2(2*nt);           % Fourier transform points
% fIQ=fftshift(fft(sIQ,nf,3),3); % frequency of sIQHP
% 
% fre = params.CCFR;
% axis_tmp = -fre/2:fre/nf:fre/2-fre/nf;
% ftmp = (squeeze(fIQ(10,220,:)));
% figure(100);
% subplot(311);
% plot(axis_tmp, abs(ftmp));
% grid on;
% xlabel("frequency Hz");
% ylabel("abs")
% title("PDI")
% subplot(312);
% plot(axis_tmp, real(ftmp));
% grid on;
% xlabel("frequency Hz");
% ylabel("real")
% subplot(313);
% plot(axis_tmp, imag(ftmp));
% grid on;
% xlabel("frequency Hz");
% ylabel("imag")


% [xROI, zROI]=ginput(10);
% xROI=floor(xROI);
% zROI=floor(zROI);
% Mask=roipoly(PDI(:,:,3),xROI,zROI);
% figure;imagesc(Mask)

%%
IQ = struct("data",[]);
IQgroup_temp = zeros([130,385,step]);
f_num = 54000;
cnt = 1;
for i = 0:f_num-1

if i < 43200
    tmp = load(strcat(subdataset_path,num2str(i+1),".mat"),'label');
else
    tmp = load(strcat(test_sub_path,"\",num2str(i+1),".mat"),'label');
end

IQgroup_temp(:,:,mod(i,step)+1)=tmp(1).label;
if mod(i,step)+1 == step
    tmp = sIQ2PDI(IQgroup_temp);
    IQ(cnt).data = tmp(:,:,3);
    cnt = cnt+1;
    IQgroup_temp = zeros([130,385,step]);
    disp(i);
end
end

% 播放
myVideo=VideoWriter(data_path+"filtered_sIQ"); % 指定视频文件名
myVideo.FrameRate = 18000/step;
open(myVideo); % open开始写入

for i = 1:length(IQ)
    frame = IQ(i).data;
    figure(1000);
    imagesc(squeeze(20*log10(abs(frame))));
    %caxis([2.3 4])
    colormap turbo;
    colorbar;
    %title(""+num2str(i)+"/"+num2str(length(IQ)));
    disp(num2str(i)+"/"+num2str(length(IQ)));
    %pause(step/18000);
    Fr=getframe(gcf); % 抓取图窗
    writeVideo(myVideo,Fr); % 写入文件
end
close(myVideo);


%%
figure(33)
for i = 1:length(IQ)
    frame = IQ(i).data;
    figure(1000);
    imagesc(squeeze(20*log10(abs(frame))))
    %caxis([2.3 4])
    colormap bone
    title(i/length(IQ))
    pause(step/18000)
end
%V0-A90
% 23-25舒展;26-27顶部;28-29收缩

% V0-A0
% 5-6快速收缩期；15-18快速舒张期
% 1-4；7-12缓慢期
% 根据对比，血管壁不动的时候，血流造成的信号变化会被记录成位移的一部分,一种替代方案是
% 用帧与帧之间的互相关识别出脉搏波速度，然后根据识别出的脉搏波速度反推


