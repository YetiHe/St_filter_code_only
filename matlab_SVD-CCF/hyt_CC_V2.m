function total_displace_interp = hyt_CC_V2(current_frame,next_frame,current_RF_frame,next_RF_frame,...
    coarse_A_window_number,accuracy_A_window_number,accuracy_step_occupy,coarse_resolution,...
    accurate_resolution, mask)
%HYT_CC_V2 此处显示有关此函数的摘要
%% 先评估轴向位移 ——> 线性插值轴向
% 应该先轴向粗+精位移，再横向粗+精位移，
% current_frame =double(abs(IQ1(:,:,1)));%abs(mean(IQ1(:,:,1:round(size(IQ1,3)/2)),3)); %abs(IQ1(:,:,1));
% next_frame = double(abs(IQ1(:,:,2)));%abs(mean(IQ1(:,:,round(size(IQ1,3)/2)+1:end),3));%abs(IQ2(:,:,2));
% current_RF_frame = double(RF1(:,:,1));%mean(RF1(:,:,1:round(size(IQ1,3)/2)),3);%RF1(:,:,1);
% next_RF_frame = double(RF1(:,:,2));%mean(RF1(:,:,round(size(IQ1,3)/2)+1:end),3);%RF2(:,:,2);
% coarse_A_window_number = 8; % [x,y], i.e.the number of the window along the both directions
% accuracy_A_window_number = 4; % [x,y], i.e.the number of the window along both directions
% accuracy_step_occupy = 0.75; % overlap ratio
% %current_Box_loc = Box_loc;
% coarse_resolution = [dx,dz*iq_decimation_factor]; %unit: m;
% accurate_resolution = dz; % unit: m
% %resolution = [PRSSinfo.FWHM(1),PRSSinfo.FWHM(3)]; % x z unit: m
% sampling_fre = P.CCFR; %P.sampling*1e6; % HZ
% formal
if isempty(mask) == 1
    mask = ones(size(current_frame));
end
current_frame = current_frame.*mask;
next_frame = next_frame.*mask;
current_RF_frame = current_RF_frame.*mask;
next_RF_frame = next_RF_frame.*mask;

% 批范式化，
IQ_cat = cat(2,current_frame,next_frame);
IQ_mean = mean(IQ_cat(:));
IQ_std = std(IQ_cat(:));
current_frame = (current_frame - IQ_mean)/IQ_std;
next_frame = (next_frame - IQ_mean)/IQ_std;

RF_cat = cat(2,current_RF_frame,next_RF_frame);
RF_mean = mean(RF_cat(:));
RF_std = std(RF_cat(:));
current_RF_frame = (current_RF_frame - RF_mean)/RF_std;
next_RF_frame = (next_RF_frame - RF_mean)/RF_std;


median_size = [3,3];
[nz,nch] = size(current_frame);
coarse_window = ceil(nz/coarse_A_window_number);
Bline_factor = ceil(nz/nch);
accu_factor = 5;
% 
% next_frame_interp = imresize(next_frame,[size(next_frame,1),size(next_frame,2)*Bline_factor],"Method","bilinear");

% 粗估计, 先估计轴向位移，偏置轴向位置后再单独估计插值后的高精度横向位移；之后精估计阶段可能不会再估计横向位移。
coarse_displace_pixel = zeros([coarse_A_window_number,nch]);
parfor ich = 1:nch
    aline = current_frame(:,ich);
    aline_next = next_frame(:,ich);
    for isegment = 1:coarse_A_window_number
        segment_start = (isegment-1)*coarse_window+1;
        segment_end = min(coarse_window*isegment,nz);
        template = aline(segment_start:segment_end);
        Aline_displace_record = zeros([1,nz-length(template)+1]);
        for j = 1:nz-length(template)+1
            temp_map = aline_next(j:j+length(template)-1);
            temp = xcorr(template,temp_map,0,"normalized");
            Aline_displace_record(j) = temp;
        end
        temp = find(Aline_displace_record == max(Aline_displace_record));
        dA = temp(1) - segment_start;
        
        aline_next_lateral = next_frame(max(1,segment_start+dA):min(segment_end+dA,size(next_frame,1)),:);
        if length(template) > length(aline_next_lateral)
            dB = 0; % 边缘的就不比了
        else
            Bline_displace_record = zeros([1,nch]);
            for j = 1:nch
                temp_map = aline_next_lateral(:,j);
                temp = xcorr(template,temp_map,0,"normalized");
                Bline_displace_record(j) = temp;
            end
            accu_axis = [1:1/Bline_factor:length(Bline_displace_record)];
            Bline_displace_interp = interp1([1:length(Bline_displace_record)],Bline_displace_record,accu_axis,"linear");
            temp = find(Bline_displace_interp == max(Bline_displace_interp(:)));
            dB =  accu_axis(temp(1)) - ich;
        end
        
        coarse_displace_pixel(isegment,ich) = dB + 1i*dA; % record pixels, [x,y] 即 [dx,dz]
    end
end
mask2 = imresize(mask,size(coarse_displace_pixel),"nearest"); % 缩小2值图
coarse_displace_pixel = coarse_displace_pixel.*mask2; 

coarse_displace_interp = imresize(real(coarse_displace_pixel)*coarse_resolution(1),[nz,nch],"method","bilinear")...
    +1i*imresize(imag(coarse_displace_pixel)*coarse_resolution(2),[nz,nch],"method","bilinear");
coarse_displace_interp = coarse_displace_interp.*mask;

coarse_displace_interp = medfilt2(real(coarse_displace_interp),median_size)...
        + 1i*medfilt2(imag(coarse_displace_interp),median_size);
coarse_displace_interp = coarse_displace_interp.*mask;

% 精位移只针对轴向
total_displace_interp = zeros(size(coarse_displace_interp));
accuracy_window_sub_factor = 1+(accuracy_A_window_number-1)*(1-accuracy_step_occupy);
for ich = 1:nch
    aline = current_RF_frame(:,ich);
    aline_next = next_RF_frame(:,ich);
    for isegment = 1:coarse_A_window_number % 粗位移的窗口，内部计算的是
        segment_start = (isegment-1)*coarse_window+1;
        segment_end = min(coarse_window*isegment,nz);
        template = aline(segment_start:segment_end);
        aline_next_segment = aline_next(segment_start+imag(coarse_displace_pixel(isegment,ich)):segment_end+imag(coarse_displace_pixel(isegment,ich)));

        sub_window_length = ceil(length(template)/accuracy_window_sub_factor);
        step = ceil(length(template)*(1-accuracy_step_occupy));
        accu_displace_map = zeros([accuracy_A_window_number,1]);
        for accu_segment = 1:accuracy_A_window_number
            accu_segment_start = (accu_segment-1)*step+1;
            accu_segment_end = min((accu_segment-1)*step+sub_window_length,length(template));
            sub_template = template(accu_segment_start:accu_segment_end);
            Aline_displace_record = zeros([1,length(template)-sub_window_length+1]);
            for j = 1:length(template)-length(sub_template)+1
                temp_map = aline_next_segment(j:j+length(sub_template)-1);
                temp = xcorr(sub_template,temp_map,0,"normalized");
                Aline_displace_record(j) = temp;
            end
            accu_axis = [1:1/accu_factor:length(Aline_displace_record)];
            Aline_displace_interp = interp1([1:length(Aline_displace_record)],Aline_displace_record,accu_axis,"cubic");
            temp = find(Aline_displace_interp == max(Aline_displace_interp(:)));
            dA =  accu_axis(temp(1)) - accu_segment_start;
            %size(sub_template)
            accu_displace_map(accu_segment) = dA;
        end

        accu_displace_map_interp = imresize(accu_displace_map*accurate_resolution,[1,length(template)],"Method","cubic");
        total_displace_interp(segment_start:segment_end,ich) = coarse_displace_interp(segment_start:segment_end,ich)+1i*accu_displace_map_interp';
    end
end
total_displace_interp = total_displace_interp.*mask;
end

