function [mid_d,max_d,fast_mid_d, slow_mid_d] = hyt_statistc_for_CC(img,b_map)
% 预计加入血管壁注意力机制，则只针对轴向
%  img为输入的位移图，记录的数据的单位需要为真实物理单位，不能是像素。
%  输出为计算的整体位移，输出单位与输入的单位相同，一共三种表征：
% mid_d: 中位数（因为组织位移速度分布极度不均衡，因此计算均值没有意义，遂用中位数表示）
% max_d: 最大值（参考价值低）
% fast_mid_d: 速度分布中较快那一半数据的中位数，物理意义是：可能是血管壁中心，或许更有利于代表血管壁的移动
if isempty(b_map)
    b_map = ones(size(img));
end
img = abs(img).*b_map;  % 只关注幅度，因为正负是移动方向
cal_temp2 = img(img>0); % 我们只关注组织的运动速度，所以只取动的部分的幅度均值

if isempty(cal_temp2) %说明没在动
    cal_temp2 = 0;
end
mid_d = prctile(cal_temp2,50);  % 因为速度分布极度不均匀，取mean没有意义，此处取中位数以代表整体移速

thre1 = prctile(cal_temp2,[70,80]); % 计算高速区的均值，去掉最大的几个值，因为很可能是边缘效应导致的噪音
thre2 = prctile(cal_temp2,[20,30]);
%cal_temp3 = cal_temp2(cal_temp2 >= thre(1) & cal_temp2 <= thre(2));
%cal_temp4 = cal_temp2(cal_temp2 <= thre(1));
cal_temp3 = cal_temp2(cal_temp2 >= thre1(1) & cal_temp2 <= thre1(2));
cal_temp4 = cal_temp2(cal_temp2 >= thre2(1) & cal_temp2 <= thre2(2));

fast_mid_d = mean(cal_temp3(:)); %prctile(cal_temp3,50);%(max(cal_temp3)-min(cal_temp3))/2 + min(cal_temp3) %
slow_mid_d = mean(cal_temp4(:));
% slow_mid_d = prctile(cal_temp2,25);
max_d = max(cal_temp3); % 要幅度最大对应的数据，但可能是噪音，参考价值不大
end

