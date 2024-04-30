import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import math
import scipy.io
import torch
from torchvision import transforms
import h5py
import cv2
import tools



'''
@param:
    dataset_dir: the dir of the whole dataset
    mode: "train" or "test" or "" (no separation dir in train and test)
    label_type: "regression" or "segmentation"
    begin_index: the index of the beginning sample
    number: the number of samples to load. If number is -1, then load all the samples behind the begin_index
'''
# 看看能不能一次性载入所有子数据集然后对于每个子数据集的片段循环遍历，目前只支持等数据量的多子数据集
class DataSet3D_all(Dataset):
    # 输入地址
    def __init__(self, ori_dataset_dir,  mode="train", label_type="regression",
                 dim=[64, 176, 200], sequence_step=200):
        self.dataset_dir = ori_dataset_dir
        self.mode = mode
        self.label_type = label_type
        self.dim = dim
        self.sequence_step = max(sequence_step, 1)
        self.sub_dataset_list = next(os.walk(self.dataset_dir))[1]
        # print(self.sub_dataset_list)

    def __getitem__(self, index):
        #  按照dim的前2个维度范式化，然后按照dim第三个维度载入相应数量图片拼成3维样本
        #  每次getitem的时候，按照step来跳过相应2维样本
        #  这样的话得要关掉shuffle，而且得要确保载入的样本输入同一序列（一共20个不同序列）
        #  建议20个序列设置为20个子数据集，在外部控制切换dataset与dataloader（已完成）
        #  每个index都是按照等效数据集的索引来的，不是真实数据集的索引。参考__len__()的重写所定义的总长度，所以取的时候记得乘步长
        sub_dataset_id = (index + 1) % len(self.sub_dataset_list) - 1 # 假设有5个等数据量的子数据集，则相对应出来0 1 2 3 -1，记得左移一位
        if sub_dataset_id < 0:
            sub_dataset_id = len(self.sub_dataset_list) - 1
        file_name_list = self.__load_file_name_list__(sub_dataset_id)
        # print(file_name_list)
        sub_index = int(np.floor(index / len(self.sub_dataset_list)))
        print(str(index) + ": " + self.sub_dataset_list[sub_dataset_id] + ": " + str(sub_index))

        begin_index = sub_index * self.sequence_step
        sequence = self.dim[2]
        samples_list = file_name_list[begin_index:begin_index+sequence]
        standard_size = [self.dim[0], self.dim[1]]
        samples = np.zeros([2, self.dim[0], self.dim[1], self.dim[2]])  # 这里改了
        labels = np.zeros([2, self.dim[0], self.dim[1], self.dim[2]])
        masks = np.zeros([1, self.dim[0], self.dim[1], self.dim[2]])
        cnt = 0
        # print("get!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        for sample_name in samples_list:
            sample_dir = os.path.join(self.dataset_dir, self.sub_dataset_list[sub_dataset_id], self.mode, sample_name)
            # print(sample_dir)
            mat_data = scipy.io.loadmat(sample_dir)
            sp = mat_data['sample']
            lb = mat_data['label']
            # 范式化
            if self.label_type == "regression":
                # sp, lb = self.__normalize_with_label__(sp, lb)

                sub_sample_real = cv2.resize(np.real(sp), [standard_size[1], standard_size[0]])# cv2.resize的雷点：目标维度是[x,y]排列
                # print(sub_sample_real.shape)
                sub_sample_imag = cv2.resize(np.imag(sp), [standard_size[1], standard_size[0]])

                # print(sub_sample.shape)
                sub_label_real = cv2.resize(np.real(lb), [standard_size[1], standard_size[0]])
                sub_label_imag = cv2.resize(np.imag(lb), [standard_size[1], standard_size[0]])
            else:
                sub_sample_real = cv2.normalize(cv2.resize(np.real(sp), [standard_size[1], standard_size[0]]), dst=None,
                                                alpha=0, beta=1,
                                                norm_type=cv2.NORM_MINMAX)  # cv2.resize的雷点：目标维度是[x,y]排列
                # print(sub_sample_real.shape)

                sub_sample_imag = cv2.normalize(cv2.resize(np.imag(sp), [standard_size[1], standard_size[0]]), dst=None,
                                                alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                # print(sub_sample.shape)
                sub_label_real = cv2.normalize(cv2.resize(np.real(lb), [standard_size[1], standard_size[0]]), dst=None,
                                               alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                sub_label_imag = cv2.normalize(cv2.resize(np.imag(lb), [standard_size[1], standard_size[0]]), dst=None,
                                               alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


            sub_mask = cv2.resize(mat_data['mask'], [standard_size[1], standard_size[0]], interpolation=cv2.INTER_NEAREST)
            # 千万不要用numpy的resize！！！numpy的resize要不全填充0，要不给你repmat，缩放图像就是个雷！！！！
            # cv2的resize选最近邻插值就可以缩放2值图了

            samples[0, :, :, cnt] = sub_sample_real
            samples[1, :, :, cnt] = sub_sample_imag
            # samples[2, :, :, cnt] = sub_sample_gradient
            labels[0, :, :, cnt] = sub_label_real
            labels[1, :, :, cnt] = sub_label_imag
            masks[0, :, :, cnt] = sub_mask
            cnt = cnt + 1

        samples = torch.from_numpy(samples)
        labels = torch.from_numpy(labels)
        masks = torch.from_numpy(masks)
        samples = samples.to(torch.float32)
        labels = labels.to(torch.float32)
        masks = masks.to(torch.float32)
        return samples, labels, masks

    def __len__(self):
        num = 0
        for i in range(len(self.sub_dataset_list)):
            temp_file_list = self.__load_file_name_list__(i)
            num = num + (math.floor((len(temp_file_list) - self.dim[2])/self.sequence_step)+1)
        return num

    def __standardize_time__(self, x):
        mean_x = np.tile(np.mean(x, axis=2).reshape([x.shape[0], x.shape[1], 1]), reps=[1, 1, x.shape[2]])
        std_x = np.tile(np.std(x, axis=2).reshape([x.shape[0], x.shape[1], 1]), reps=[1, 1, x.shape[2]])
        x_final = (x - mean_x)/std_x
        return x_final

    def __normalize_time__(self, x):
        min_x = np.tile(np.min(x, axis=2).reshape([x.shape[0], x.shape[1], 1]), reps=[1, 1, x.shape[2]])
        max_x = np.tile(np.max(x, axis=2).reshape([x.shape[0], x.shape[1], 1]), reps=[1, 1, x.shape[2]])
        x_final = (x - min_x)/(max_x - min_x)
        return x_final

    def __normalize_with_label__(self, s, l):
        mean_s, mean_l = np.mean(s), np.mean(l)
        std_s, std_l = np.std(s), np.std(l)
        mean_factor = (mean_l + mean_s) / 2
        std_factor = (std_s + std_l) / 2
        s = (s - mean_factor)/std_factor
        l = (l - mean_factor)/std_factor
        return s, l

    # to load the file name list and support the self-defined slice
    def __load_file_name_list__(self, id):
        sub_dataset_name = self.sub_dataset_list[id]
        # print(sub_dataset_name)
        all_list = os.listdir(os.path.join(self.dataset_dir, sub_dataset_name, self.mode))
        # print(os.path.join(self.dataset_dir, sub_dataset_name, self.mode))
        all_list.sort(key=lambda x: int(x.split('.')[0]))  # 排序，因此要求保存的文件名必须为代表序列顺序的纯数字
        return all_list


if __name__ == "__main__":
    train_ds = DataSet3D_all(".\\data", dim=[64, 180, 200], sequence_step=1)
    # 定义数据加载
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    for i, (ct, label, mask) in enumerate(train_dl):
        print(i, ct.dtype, ct.shape, label.dtype, label.shape, mask.dtype, mask.shape)

    # f = h5py.File(".\\testt.h5", "r")
    # ff = h5py.File(".\\tt.h5", "w")
    # for key in f.keys():
    #     print(f[key].name)
    #     tmp_sample = f[key][:]
    #     print(tmp_sample.shape)
    #     if f[key].name == "/iIQ":
#
    #     #    ff.create_dataset("/data", data=f[key][:])
    # ff.close()
    # f.close()
