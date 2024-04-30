import torch
import math
import cv2
import os
import time
import json
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat
# import pytorch_ssim
from skimage.metrics import structural_similarity
import model

LAST_CHECKPOINT_NAME = "last_checkpoint"  # The defined name of the last checkpoint
DEFAULT_SAVE_DIR = ".\\record"  # The default save dir
SMOOTH_FACTOR = 1E-8
SSIM_WIN_SIZE = 9



# create the file
def mkdir(path):  # ensure whether there exists the specific file
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


# save the checkpoint of the training
def save_checkpoint(state, path, filename='temp'):  # 存入当前进度，从而允许使用者随时停止，下次接着上次的进度继续
    if path == "":
        path = DEFAULT_SAVE_DIR
    mkdir(path)
    name = os.path.join(path, filename + '.pth')
    print("saving checkpoint in the \"", name, '\"...\n')
    torch.save(state, name)  # state是一个字典
    last_name = os.path.join(path, LAST_CHECKPOINT_NAME + ".pth")
    print("saving checkpoint in the \"", last_name, '\"...\n')
    torch.save(state, last_name)  # state是一个字典


# 保存当前模型参数+模型的test或validation指标+对应超参数
# save the model's parameters, the metrics, and the corresponding hyperparameters
def save_model(path, file_name, model, metrics_data, paras):
    file_name = "model-" + file_name + "-at-"+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    save_checkpoint({"model_parameters": model.state_dict(),
                     "metrics": metrics_data,
                     "hyper_parameters": paras}, path, file_name)


# 提取当前模型参数，file输入带后缀的文件名
# 返回的模型参数（需要外部载入nn.model），指标，超参
'''
@:param: 
'''
def load_hyperparameters_metrics(file):
    metric = torch.load(file)['metrics']
    hyper_para = torch.load(file)['hyper_parameters']
    return {"metrics": metric, "hyper_parameters": hyper_para}


def load_model(file, model, hyper_para):
    if hyper_para["device"] == "cuda" and not torch.cuda.is_available():
        hyper_para["device"] = "cpu"
        print("The device is not okay for gpu. It is automatically using cpu now...")
    model.to(torch.device(hyper_para["device"]))  # 传入相应设备
    model.load_state_dict(torch.load(file, map_location=str(torch.device(hyper_para["device"])))['model_parameters'])
    return model, hyper_para


# 保存超参，输入完整file名（带地址）
# save the hyperparameters in txt file so that I can check it without interacting with .pth file.
# I might not need it
def save_hyper_parameters_txt(file, paras):
    with open(file, 'w') as f:
        f.write(json.dumps(paras))


# 提取超参，file输入带后缀的文件名(.txt)
def load_hyper_parameters_txt(file):
    file = open(file, 'r')
    tmp = file.read()
    hyper_para = json.loads(tmp)
    file.close()
    return hyper_para


def show_image(img, fig_handle_id=99, ti="temp", save_dir=DEFAULT_SAVE_DIR):
    img = np.asarray(img)
    # img = 255 * (img - np.min(img))/(np.max(img) - np.min(img))
    plt.figure(fig_handle_id)
    plt.imshow(img)
    plt.colormaps()
    plt.colorbar()
    plt.title(ti)
    plt.show(block=False)
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, ti + "_tmp_fig.png"))
    # plt.pause(3)
    plt.close(fig_handle_id)


def transform_binary_image(seq, size5D=[6,1,96,96,200], ti="", save_dir=DEFAULT_SAVE_DIR, select_batch=0):
    with torch.no_grad():
        tmp = torch.zeros(seq.shape)
        tmp[seq >= 0.5] = 1
        data = np.asarray(torch.reshape(tmp, shape=size5D))
        img = data[select_batch, 0, :, :, 0]
        show_image(img=img, fig_handle_id=41, ti=ti, save_dir=save_dir)
    return img


def transform_normal_image(data, ti="", save_dir=DEFAULT_SAVE_DIR):
    # 默认传入的维度为：通道，高，宽，时间（不包含batch）
    with torch.no_grad():
        data = np.asarray(data)
        data_complex = np.vectorize(complex)(data[0, :, :, :], data[1, :, :, :])
        PDI_img = np.abs(data_complex) ** 2
        PDI_ave = np.mean(PDI_img, axis=2)  # PDI_img[:, :, 0]
        # PDI_show = (PDI_ave - np.min(PDI_ave)) / (np.max(PDI_ave) - np.min(PDI_ave))
        PDI_show = 10 * np.log10(PDI_ave)  # in dB
        show_image(img=PDI_show, fig_handle_id=41, ti="dB-" + ti, save_dir=save_dir)
        show_image(img=PDI_ave, fig_handle_id=41, ti=ti, save_dir=save_dir)
    return data_complex


def transform_fft_image(data, ti="", save_dir = DEFAULT_SAVE_DIR):
    # 默认传入的维度为：通道，高，宽，时间（不包含batch）
    with torch.no_grad():
        data = np.asarray(data)
        data_complex = np.vectorize(complex)(data[0, :, :, :], data[1, :, :, :])
        data_img = np.fft.ifft(np.fft.ifftshift(data_complex), axis=2)
        PDI_img = np.abs(data_img)**2
        PDI_show = np.mean(PDI_img, axis=2)
        show_image(img=PDI_show, fig_handle_id=41, ti=ti, save_dir=save_dir)


def normalize_max_min_for_2_5D_data(img):
    assert len(img.shape) == 5
    # ！！！！注意阈值分割的时候，分批分时间单独划分阈值！！！(阈值是个2维张量)
    # 先范式化到同一尺度
    max_factor = torch.tile(
        (torch.max(torch.max(img, dim=2)[0], dim=2)[0]).view([img.shape[0], img.shape[1], 1, 1, img.shape[4]]),
        dims=[1, 1, img.shape[2], img.shape[3], 1])
    min_factor = torch.tile(
        (torch.min(torch.min(img, dim=2)[0], dim=2)[0]).view([img.shape[0], img.shape[1], 1, 1, img.shape[4]]),
        dims=[1, 1, img.shape[2], img.shape[3], 1])
    img_norm = (img - min_factor) / (max_factor - min_factor + SMOOTH_FACTOR)
    return img_norm


def calculate_US_score(img, masks=None):
    # 除了CNR和SNR之外
    # 拿到IQ的abs图, 记得维度：批，通道，高，长，时间，变为：批，高，长，时间
    assert len(img.shape) == 5
    # ！！！！注意阈值分割的时候，分批分时间单独划分阈值！！！(阈值是个2维张量)
    # 计算PDI

    img_PDI = (torch.pow(torch.squeeze(img[:, 0, :, :, :]), 2.0)
               + torch.pow(torch.squeeze(img[:, 1, :, :, :]), 2.0)).cpu()

    if masks is not None:
        masks = torch.squeeze(masks, dim=1)

    if len(img_PDI.shape) != 4:
        img_PDI = torch.unsqueeze(img_PDI, dim=0)
    # show_image(img_PDI[0, :, :, 0], ti="111")

    # 计算PDI的高低阈值并制作列表
    fore_mask = torch.zeros(img_PDI.shape)
    back_mask = torch.zeros(img_PDI.shape)

    if masks is None:  # 无掩码，只是自己内部做的snr
        thre_fore_ori = torch.mean(img_PDI, dim=[1, 2]) + 0.5 * torch.std(img_PDI, dim=[1, 2])
        thre_fore = torch.tile(thre_fore_ori.view([img_PDI.shape[0], 1, 1, img_PDI.shape[3]]),
                               dims=[1, img_PDI.shape[1], img_PDI.shape[2], 1])
        thre_back_ori = torch.mean(img_PDI, dim=[1, 2]) - 0.8 * torch.std(img_PDI, dim=[1, 2])
        thre_back = torch.tile(thre_back_ori.view([img_PDI.shape[0], 1, 1, img_PDI.shape[3]]),
                               dims=[1, img_PDI.shape[1], img_PDI.shape[2], 1])
        fore_mask[img_PDI >= thre_fore] = 1
        back_mask[img_PDI < thre_back] = 1
    else:  # 提供前景和背景信息，snr与cnr都要在掩码内做
        fore_mask[masks == 1] = 1
        back_mask[masks == 0] = 1

    img_fore = (torch.multiply(fore_mask, img_PDI)).view(-1)
    list_fore = img_fore[img_fore > 0]
    img_back = (torch.multiply(back_mask, img_PDI)).view(-1)
    list_back = img_back[img_back > 0]

    # SNR = 10log10(mean(foreground's PDI) / std(background's PDI) )
    snr = (torch.mean(list_fore) / (torch.std(list_back) + SMOOTH_FACTOR)).item()

    # CNR, 非分贝表示，否则会引入负数，没有意义
    cnr = (torch.abs(torch.mean(list_fore) - torch.mean(list_back)) / (torch.std(list_back) + SMOOTH_FACTOR)).item()
    return [snr, cnr]


def calculate_US_regression_ratio(predict_ori, label_ori, masks):
    # 计算针对超声成像的评价指标，SNR与CNR; 该函数只有回归模型会去用，predict和label都是图像，非序列
    # 要不要考虑用图像处理常用的图像间的衡量指标，如PSNR, SSIM
    with torch.no_grad():
        if len(predict_ori.shape) != 5:
            predict_ori = torch.unsqueeze(predict_ori, dim=0)
            label_ori = torch.unsqueeze(label_ori, dim=0)
            masks = torch.unsqueeze(masks, dim=0)
        criterion = torch.nn.modules.loss.MSELoss(reduction='sum')
        loss = criterion(predict_ori, label_ori)
        MSE_loss = loss.item() / (label_ori.view(-1)).shape[0]
        criterion = torch.nn.modules.loss.L1Loss(reduction='sum')
        loss = criterion(predict_ori, label_ori)
        MAE_loss = loss.item() / (label_ori.view(-1)).shape[0]
        criterion = torch.nn.modules.loss.SmoothL1Loss(reduction='sum')
        loss = criterion(predict_ori, label_ori)
        Huber_loss = loss.item() / (label_ori.view(-1)).shape[0]
        criterion = model.hyt_SSIM_loss()
        loss = criterion(predict_ori, label_ori)
        ssim_loss = loss.item()

        predict = normalize_max_min_for_2_5D_data(predict_ori)
        label = normalize_max_min_for_2_5D_data(label_ori)

        predict_set = calculate_US_score(predict, masks)
        label_set = calculate_US_score(label, masks)

        snr_error = (label_set[0] - predict_set[0]) / (label_set[0] + SMOOTH_FACTOR)
        cnr_error = (label_set[1] - predict_set[1]) / (label_set[1] + SMOOTH_FACTOR)

        predict_abs = torch.sqrt(((torch.pow(torch.squeeze(predict_ori[:, 0, :, :, :]), 2.0)
                                 + torch.pow(torch.squeeze(predict_ori[:, 1, :, :, :]), 2.0)))).cpu()
        label_abs = torch.sqrt(((torch.pow(torch.squeeze(label_ori[:, 0, :, :, :]), 2.0)
                               + torch.pow(torch.squeeze(label_ori[:, 1, :, :, :]), 2.0)))).cpu()

        if len(predict_abs.shape) != 4:
            predict_abs = torch.unsqueeze(predict_abs, dim=0)
            label_abs = torch.unsqueeze(label_abs, dim=0)

        # 调整深度，与batch叠加到一起，成为类似通道的概念，原：批，高，长，时间 -> 批，时间，高，长 -> 批*时间，高，长
        predict_abs_per = torch.permute(predict_abs, dims=[0, 3, 1, 2])
        label_abs_per = torch.permute(label_abs, dims=[0, 3, 1, 2])

        predict_abs_ssim = torch.reshape(predict_abs_per, shape=[predict_abs_per.shape[0] * predict_abs_per.shape[1],
                                                                 predict_abs_per.shape[2], predict_abs_per.shape[3]])
        label_abs_ssim = torch.reshape(label_abs_per, shape=[label_abs_per.shape[0] * label_abs_per.shape[1],
                                                             label_abs_per.shape[2], label_abs_per.shape[3]])

        ssim_ave = structural_similarity(im1=np.asarray(label_abs_ssim), im2=np.asarray(predict_abs_ssim),
                                         channel_axis=0, win_size=SSIM_WIN_SIZE,
                                         data_range=max(torch.max(predict_abs).item(), torch.max(label_abs).item()))
    return [snr_error, cnr_error, ssim_ave, MSE_loss, MAE_loss, Huber_loss, ssim_loss]


def calculate_confuse_based_score(confuse):
    # confuse: [TP, TN, FP, FN]
    confuse = np.asarray(confuse)
    if len(confuse.shape) == 1:
        TP = confuse[0]
        TN = confuse[1]
        FP = confuse[2]
        FN = confuse[3]
    else:
        TP = confuse[:, 0]
        TN = confuse[:, 1]
        FP = confuse[:, 2]
        FN = confuse[:, 3]
    dice_score = 2 * TP / (2 * TP + FN + FP + SMOOTH_FACTOR)
    accuracy_score = (TP + TN) / (TP + FN + TN + FP + SMOOTH_FACTOR)
    IOU = TP / (TP + FN + FP + SMOOTH_FACTOR)
    if len(confuse.shape) == 1:
        return [accuracy_score, dice_score, IOU]
    else:
        result = np.asarray([list(accuracy_score), list(dice_score), list(IOU)])
        result = result.transpose()
        return list(result)


def calculate_confuse_matrix_score(predict, label):
    # 用于评价图像分割的基于混淆矩阵做的各种score目前输出
    assert(len(predict.shape) == len(label.shape))
    assert(predict.shape[0] == label.shape[0])
    # sm = torch.nn.Sigmoid()  # 如果用的是BCEwithlogits，则需要自己映射一下sigmod
    # predict = sm(predict)
    with torch.no_grad():
        label_mask = label.cpu()
        predict_mask = torch.zeros(predict.shape)
        predict_mask[predict >= 0.5] = 1
        predict_mask[predict < 0.5] = 0

        tmp_mask = label_mask + predict_mask
        TP = (tmp_mask[tmp_mask == 2]).shape[0]
        TN = (tmp_mask[tmp_mask == 0]).shape[0]
        tmp_mask[tmp_mask == 2] = 0
        label_different_mask = tmp_mask * label_mask
        predict_different_mask = tmp_mask * predict_mask
        FN = (label_different_mask[label_different_mask == 1]).shape[0]
        FP = (predict_different_mask[predict_different_mask == 1]).shape[0]

    return [TP, TN, FP, FN]


# 评估表现，可能需要做成一个metric类，或直接返回一个metric字典
class Metrics:
    def __init__(self, loss_name="SmoothL1Loss"):
        self.data = {"loss_name": loss_name,
                     "mean_train_loss": [],
                     "mean_validate_loss": [],
                     "mean_special_train_score": [],
                     "mean_special_validate_score": []}  #
        self.best_loss = 1

    def reset(self):
        for key in self.data:
            self.data[key] = 0

    # 根据平均loss来决定保存具有最优validate的那个模型;如果没有validate数据就参考train数据
    def is_the_best_for_now(self):
        if len(self.data["mean_validate_loss"]) >= 3:  # 至少3轮validate才可行信
            tmp = np.asarray(self.data["mean_validate_loss"])
            mean_validate_loss_vals = tmp[2:tmp.shape[0]]
            # 记得np用的是matlab语法，切片的时候索引不从0开始，但是取数组单个元素的时候索引按照python语法
            if np.min(mean_validate_loss_vals) == mean_validate_loss_vals[mean_validate_loss_vals.shape[0]-1]:
                return True
            else:
                return False
        else:
            mean_train_loss_vals = self.data["mean_train_loss"]
            if min(mean_train_loss_vals) == mean_train_loss_vals[len(mean_train_loss_vals)-1]:
                return True
            else:
                return False

    # 一个epoch调用一次，自动计入该epoch内平均loss和最大loss
    def put_loss(self, vals, is_train):
        if len(vals) == 0:
            return
        val = sum(vals) / len(vals)
        if is_train == 1:
            self.data["mean_train_loss"].append(val)
        else:
            self.data["mean_validate_loss"].append(val)

    # 一个epoch调用一次，自动计入该epoch内平均score和最大score
    def put_score(self, ori_vals, is_train):
        vals = np.asarray(ori_vals)
        if vals.shape[0] == 0:
            return
        val = vals.sum(0) / vals.shape[0]
        if is_train == 1:
            self.data["mean_special_train_score"].append(list(val))
        else:
            self.data["mean_special_validate_score"].append(list(val))

    # 根据已有loss数据画出mean和max的loss-epoch变化图，
    def draw_loss_line(self, save_dir=DEFAULT_SAVE_DIR, fig_handle_id=99, loss_name=""):
        if loss_name == "":
            loss_name = self.data["loss_name"]
        mean_train_loss_vals = np.asarray(self.data["mean_train_loss"])
        if len(self.data["mean_validate_loss"]) != 0:
            mean_validate_loss_vals = np.asarray(self.data["mean_validate_loss"])
        epoch_axis = []
        for i in range(0, len(mean_train_loss_vals)):
            epoch_axis.append(i + 1)
        epoch_axis = np.asarray(epoch_axis)

        plt.figure(fig_handle_id)
        plt.plot(epoch_axis, mean_train_loss_vals, color='red', label="averaged train "+loss_name)
        if len(self.data["mean_validate_loss"]) != 0:
            plt.plot(epoch_axis, mean_validate_loss_vals, color='blue', label="averaged validate "+loss_name)
        plt.xticks(range(int(epoch_axis[0]), int(epoch_axis[len(mean_train_loss_vals) - 1]) + 1))
        plt.xlabel("epoch")
        plt.ylabel(loss_name)
        plt.grid(visible=True)
        plt.title("averaged " + loss_name)
        plt.legend()

        plt.show(block=False)
        mkdir(save_dir)
        plt.savefig(os.path.join(save_dir, "loss-epoch_tmp_fig.png"))
        plt.pause(3)
        plt.close(fig_handle_id)

        # 根据已有loss数据画出mean和max的loss-epoch变化图，
    def draw_score_line(self, save_dir=DEFAULT_SAVE_DIR, fig_handle_id=99, name_info="SNR", special_index=0,
                        is_confuse_matrix=False):
        if is_confuse_matrix == False:
            mean_train_loss_vals = np.asarray(self.data["mean_special_train_score"])[:, special_index]
            if len(self.data["mean_special_validate_score"]) != 0:
                mean_validate_loss_vals = np.asarray(self.data["mean_special_validate_score"])[:, special_index]
            self.__draw_score_line__(fig_handle_id=fig_handle_id, mean_train_loss_vals=mean_train_loss_vals,
                                     mean_validate_loss_vals=mean_validate_loss_vals, name_info=name_info,
                                     save_dir=save_dir)
        else:
            mean_train_loss_vals = np.asarray(calculate_confuse_based_score(self.data["mean_special_train_score"]))[:, special_index]
            if len(self.data["mean_special_validate_score"]) != 0:
                mean_validate_loss_vals = np.asarray(calculate_confuse_based_score(self.data["mean_special_validate_score"]))[:, special_index]
            self.__draw_score_line__(fig_handle_id=fig_handle_id, mean_train_loss_vals=mean_train_loss_vals,
                                     mean_validate_loss_vals=mean_validate_loss_vals, name_info=name_info,
                                     save_dir=save_dir)

    def __draw_score_line__(self, fig_handle_id, mean_train_loss_vals, mean_validate_loss_vals, name_info, save_dir):
        epoch_axis = []
        for i in range(0, len(mean_train_loss_vals)):
            epoch_axis.append(i + 1)
        epoch_axis = np.asarray(epoch_axis)

        plt.figure(fig_handle_id)
        plt.plot(epoch_axis, mean_train_loss_vals, color='red', label="averaged train "+name_info)
        if len(self.data["mean_special_validate_score"]) != 0:
            plt.plot(epoch_axis, mean_validate_loss_vals, color='blue', label="averaged validate "+name_info)
        plt.xticks(range(int(epoch_axis[0]), int(epoch_axis[len(mean_train_loss_vals)-1]) + 1))
        plt.title("averaged " + name_info)
        plt.xlabel("epoch")
        plt.ylabel(name_info)
        plt.grid(visible=True)
        plt.legend()

        plt.show(block=False)
        mkdir(save_dir)
        plt.savefig(os.path.join(save_dir, name_info + "-epoch_tmp_fig.png"))
        plt.pause(3)
        plt.close(fig_handle_id)


if __name__ == "__main__":
    epoch_axis = np.asarray([1,2,3])
    mean_train_loss_vals = np.asarray([3,4,63])
    name_info = "444"
    plt.figure(1)
    plt.plot(epoch_axis, mean_train_loss_vals, color='red', label="averaged train " + name_info)
    # if len(self.data["mean_special_validate_score"]) != 0:
    #     plt.plot(epoch_axis, mean_validate_loss_vals, color='blue', label="averaged validate " + name_info)
    plt.xticks(range(int(epoch_axis[0]), int(epoch_axis[3 - 1]) + 1))
    plt.title("averaged " + name_info)
    plt.xlabel("epoch")
    plt.ylabel(name_info)
    plt.grid(visible=True)
    plt.legend()

    plt.show()
    # tmp = load_hyperparameters_metrics(file_path + "\\best_checkpoint.pth")
    # me = Metrics()
    # me.data = tmp["metrics"]
    # me.draw_loss_line(save_dir=file_path, fig_handle_id=99)
    '''
    ts = torch.randn([2, 3, 4, 5])
    #ns = np.asarray(ts)
    print(ts.shape)
    thre1 = torch.mean(ts, dim=[1,2]) # torch.min(torch.min(ts,dim=1)[0],dim=1)[0]
    print(thre1.shape)
    print(thre1)
    thre = torch.tile(thre1.view([2,1,1,5]), dims=[1,3,4,1])
    print(thre.shape)
    # print(thre)
    mask = torch.zeros(ts.shape)
    mask[ts > thre] = 1
    mask[ts <= thre] = 0
    print(thre[0,:,:,0])
    print("mask")
    print(mask[0,:,:,0])
    print("ts")
    print(ts[0,:,:,0])

    lts = torch.squeeze(ts[0, :, :, 0])
    print("lts")
    print(lts)
    print(lts.shape)
    lthre = torch.mean(lts)
    print(lthre)
    lmask = torch.zeros(lts.shape)
    lmask[lts > lthre] = 1
    lmask[lts <= lthre] = 0
    print("lmask")
    print(lmask)
    result = (lmask-torch.squeeze(mask[0,:,:,0])).view(-1)
    print(result)
    print(torch.squeeze(mask[0,:,:,0]).shape)
    print(torch.sum(result))
    '''



