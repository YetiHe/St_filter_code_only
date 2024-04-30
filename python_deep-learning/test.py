import model
import numpy as np
import data_set
import train
import tools
import torch
import os
from scipy.io import savemat
import scipy
import cv2
from torch.utils.data import Dataset, DataLoader


def test_sub_dataset(checkpoint_path=".\\record\\best_checkpoint.pth", dataset_dir=".\\data", device="cuda", batch_size=2):

    tmp = tools.load_hyperparameters_metrics(file=checkpoint_path)
    hyper_parameters = tmp["hyper_parameters"]
    #  create a metric calculator
    metrics = tools.Metrics(loss_name=hyper_parameters["loss"])
    metrics.data = tmp["metrics"]

    model_type = hyper_parameters["model_type"]
    label_type = hyper_parameters["label_type"]
    if model_type == "UNet3D":
        if label_type == "regression":
            md = model.UNet3D_regression()
        else:
            md = model.UNet3D_segmentation()
    elif model_type == "UNet2D":  # 只有regression，没做他的segmentation
        if label_type == "regression":
            md = model.UNet2D_regression()
        else:
            md = model.UNet2D_segmentation()
    md, hyper_parameters = tools.load_model(
        file=checkpoint_path, model=md, hyper_para=hyper_parameters)  # 此时已经装好了模型以及设备，可以直接用
    if hyper_parameters["device"] == "cuda" and device == "cpu":
        md.to(torch.device("cpu"))
    if hyper_parameters["device"] == "cpu" and device == "cuda":
        if torch.cuda.is_available():
            md.to(torch.device("cuda"))
        else:
            print("The device is not okay for gpu. It is automatically using cpu now...")
            device = "cpu"

    md.eval()
    metrics_sub_list = dict()
    score_sub_list = dict()

    # 按照train的思路在每个sub dataset里验证，并给出最终loss值，没有epoch所以不需要画图
    if hyper_parameters["loss"] == "MSE":
        criterion = torch.nn.modules.loss.MSELoss()
    elif hyper_parameters["loss"] == "BCELoss":
        criterion = torch.nn.modules.loss.BCELoss()

    if hyper_parameters["model_type"] == "UNet3D" or hyper_parameters["model_type"] == "UNet2D":
        # print(os.path.join(self.hyper_parameters["dataset_dir"], sub_dataset_name))
        dataset = data_set.DataSet3D(dataset_dir,
                                     mode="test", label_type=hyper_parameters["label_type"],
                                     dim=[64, 176, 200], sequence_step=100,
                                     begin_index=0, number=-1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 需要根据sub dataset 单独validate，最好给出不同，subdata set 的loss值，因为很有可能模型有角度偏好性
    loss_record = []
    score_record = []
    for i, (samples, labels) in enumerate(dataloader):
        with torch.no_grad():
            if device == "cuda":
                samples = samples.cuda()
                labels = labels.cuda()
            outputs = md(samples)
            if hyper_parameters["model_type"] == "UNet3D" and hyper_parameters["label_type"] != "regression":
                labels = labels.view(-1)
                outputs = outputs.view(-1)
            loss = criterion(outputs, labels)
            if hyper_parameters["label_type"] == "regression":
                score = tools.calculate_US_regression_ratio(outputs, labels)
            else:
                score = tools.calculate_confuse_matrix_score(outputs, labels)
            # print("      loss=", loss.item())
            loss_record.append(loss.item())
            score_record.append(score)
    pass


def test_in_all(checkpoint_path=".\\record\\best_checkpoint.pth", dataset_dir=".\\data", device="cuda", batch_size=2):
    # 用dataset3D_all来把所有的test数据混在一起跑一遍，由于不只同一个视频，所以只用展示数据图就可以了，
    # 如果想保存视频（只是用于后续展示，论文里放不了视频，看个SSIM和SNR error和CNR error就够了），用test_subdataset
    pass


def generate_mat_script_2D(subdataset_name="V0-A30-PRF18K"):
    checkpoint_path = ".\\record\\2D_ablation_all\\last_checkpoint.pth"
    # checkpoint_path = ".\\record\\fine_model_ssim_all\\last_checkpoint.pth"
    data_path = ".\\phantom_data\\"+subdataset_name+"\\test"
    # data_path = ".\\fine_data2\\" + subdataset_name + "\\test"
    save_dir = ".\\fine_test_generate2D\\"+subdataset_name
    tools.mkdir(".\\fine_test_generate2D")

    device = "cuda"
    standard_size = [64, 176]
    step = 200
    sequence = step

    tools.mkdir(save_dir)

    tmp = tools.load_hyperparameters_metrics(file=checkpoint_path)
    hyper_parameters = tmp["hyper_parameters"]
    #  create a metric calculator
    metrics = tools.Metrics(loss_name=hyper_parameters["loss"])
    metrics.data = tmp["metrics"]

    model_type = hyper_parameters["model_type"]
    label_type = hyper_parameters["label_type"]
    if model_type == "UNet3D":
        if label_type == "regression":
            md = model.UNet3D_regression()
        else:
            md = model.UNet3D_segmentation()
    elif model_type == "UNet2D":  # 只有regression，没做他的segmentation
        if label_type == "regression":
            md = model.UNet2D_regression()
        else:
            md = model.UNet2D_segmentation()
    md, hyper_parameters = tools.load_model(
        file=checkpoint_path, model=md, hyper_para=hyper_parameters)  # 此时已经装好了模型以及设备，可以直接用
    if hyper_parameters["device"] == "cuda" and device == "cpu":
        md.to(torch.device("cpu"))
    if hyper_parameters["device"] == "cpu" and device == "cuda":
        if torch.cuda.is_available():
            md.to(torch.device("cuda"))
        else:
            print("The device is not okay for gpu. It is automatically using cpu now...")
            device = "cpu"
    md.eval()

    all_list = os.listdir(data_path)
    # print(os.path.join(self.dataset_dir, sub_dataset_name, self.mode))
    all_list.sort(key=lambda x: int(x.split('.')[0]))

    for istep in range(0, int(len(all_list)/step)):
        samples_list = all_list[istep*step:(istep+1)*step]
        samples = np.zeros([2, standard_size[0], standard_size[1], sequence])  # 这里改了
        labels = np.zeros([2, standard_size[0], standard_size[1], sequence])
        cnt = 0

        for name in samples_list:
            file_name = os.path.join(data_path, name)
            mat_data = scipy.io.loadmat(file_name)
            sp = mat_data['sample']
            lb = mat_data['label']
            sub_sample_real = cv2.resize(np.real(sp), [standard_size[1], standard_size[0]])  # cv2.resize的雷点：目标维度是[x,y]排列
            # print(sub_sample_real.shape)
            sub_sample_imag = cv2.resize(np.imag(sp), [standard_size[1], standard_size[0]])

            # print(sub_sample.shape)
            sub_label_real = cv2.resize(np.real(lb), [standard_size[1], standard_size[0]])
            sub_label_imag = cv2.resize(np.imag(lb), [standard_size[1], standard_size[0]])

            samples[0, :, :, cnt] = sub_sample_real
            samples[1, :, :, cnt] = sub_sample_imag
            # samples[2, :, :, cnt] = sub_sample_gradient
            labels[0, :, :, cnt] = sub_label_real
            labels[1, :, :, cnt] = sub_label_imag
            cnt = cnt + 1

        samples = torch.unsqueeze(torch.from_numpy(samples), dim=0)
        labels = torch.unsqueeze(torch.from_numpy(labels), dim=0)
        samples = samples.to(torch.float32)
        labels = labels.to(torch.float32)

        samples = samples.cuda()
        with torch.no_grad():

            outputs = torch.zeros(size=samples.shape)
            # if torch.cuda.is_available() and hyper_parameters["device"] == "cuda":
            outputs = outputs.cuda()
            for sub_id in range(0, samples.shape[4]):
                tmp1 = md(samples[:, :, :, :, sub_id])
                outputs[:, :, :, :, sub_id] = tmp1
                # sub_loss = criterion(tmp1, labels[:, :, :, :, sub_id])

            # outputs = md(samples)


            print(outputs.shape)
            outputs = outputs.cpu()

            predict_tmp = tools.transform_normal_image(data=outputs[0, :, :, :, :].cpu(),
                                                       ti="predict"+str(istep+1),
                                                       save_dir=save_dir)
            label_tmp = tools.transform_normal_image(data=labels[0, :, :, :, :].cpu(),
                                                     ti="label" + str(istep+1),
                                                     save_dir=save_dir)
            savemat(save_dir+"\\data-"+ str(istep+1) +"-output.mat",
                {"predict": predict_tmp, "label": label_tmp})
        print(str(istep+1)+" / "+str(int(len(all_list)/step)))

def generate_mat_script(subdataset_name="V0-A30-PRF18K"):
    checkpoint_path = ".\\record\\phantom_model\\best_checkpoint.pth"
    checkpoint_path = ".\\record\\fine_model_ssim_all\\last_checkpoint.pth"
    data_path = ".\\phantom_data\\"+subdataset_name+"\\test"
    data_path = ".\\fine_data2\\" + subdataset_name + "\\test"
    save_dir = ".\\fine_test_generate\\"+subdataset_name
    tools.mkdir(".\\fine_test_generate")

    device = "cuda"
    standard_size = [64, 176]
    step = 200
    sequence = step

    tools.mkdir(save_dir)

    tmp = tools.load_hyperparameters_metrics(file=checkpoint_path)
    hyper_parameters = tmp["hyper_parameters"]
    #  create a metric calculator
    metrics = tools.Metrics(loss_name=hyper_parameters["loss"])
    metrics.data = tmp["metrics"]

    model_type = hyper_parameters["model_type"]
    label_type = hyper_parameters["label_type"]
    if model_type == "UNet3D":
        if label_type == "regression":
            md = model.UNet3D_regression()
        else:
            md = model.UNet3D_segmentation()
    elif model_type == "UNet2D":  # 只有regression，没做他的segmentation
        if label_type == "regression":
            md = model.UNet2D_regression()
        else:
            md = model.UNet2D_segmentation()
    md, hyper_parameters = tools.load_model(
        file=checkpoint_path, model=md, hyper_para=hyper_parameters)  # 此时已经装好了模型以及设备，可以直接用
    if hyper_parameters["device"] == "cuda" and device == "cpu":
        md.to(torch.device("cpu"))
    if hyper_parameters["device"] == "cpu" and device == "cuda":
        if torch.cuda.is_available():
            md.to(torch.device("cuda"))
        else:
            print("The device is not okay for gpu. It is automatically using cpu now...")
            device = "cpu"
    md.eval()

    all_list = os.listdir(data_path)
    # print(os.path.join(self.dataset_dir, sub_dataset_name, self.mode))
    all_list.sort(key=lambda x: int(x.split('.')[0]))

    for istep in range(0, int(len(all_list)/step)):
        samples_list = all_list[istep*step:(istep+1)*step]
        samples = np.zeros([2, standard_size[0], standard_size[1], sequence])  # 这里改了
        labels = np.zeros([2, standard_size[0], standard_size[1], sequence])
        cnt = 0

        for name in samples_list:
            file_name = os.path.join(data_path, name)
            mat_data = scipy.io.loadmat(file_name)
            sp = mat_data['sample']
            lb = mat_data['label']
            sub_sample_real = cv2.resize(np.real(sp), [standard_size[1], standard_size[0]])  # cv2.resize的雷点：目标维度是[x,y]排列
            # print(sub_sample_real.shape)
            sub_sample_imag = cv2.resize(np.imag(sp), [standard_size[1], standard_size[0]])

            # print(sub_sample.shape)
            sub_label_real = cv2.resize(np.real(lb), [standard_size[1], standard_size[0]])
            sub_label_imag = cv2.resize(np.imag(lb), [standard_size[1], standard_size[0]])

            samples[0, :, :, cnt] = sub_sample_real
            samples[1, :, :, cnt] = sub_sample_imag
            # samples[2, :, :, cnt] = sub_sample_gradient
            labels[0, :, :, cnt] = sub_label_real
            labels[1, :, :, cnt] = sub_label_imag
            cnt = cnt + 1

        samples = torch.unsqueeze(torch.from_numpy(samples), dim=0)
        labels = torch.unsqueeze(torch.from_numpy(labels), dim=0)
        samples = samples.to(torch.float32)
        labels = labels.to(torch.float32)

        samples = samples.cuda()
        with torch.no_grad():
            outputs = md(samples)
            print(outputs.shape)
            outputs = outputs.cpu()

            predict_tmp = tools.transform_normal_image(data=outputs[0, :, :, :, :].cpu(),
                                                       ti="predict"+str(istep+1),
                                                       save_dir=save_dir)
            label_tmp = tools.transform_normal_image(data=labels[0, :, :, :, :].cpu(),
                                                     ti="label" + str(istep+1),
                                                     save_dir=save_dir)
            savemat(save_dir+"\\data-"+ str(istep+1) +"-output.mat",
                {"predict": predict_tmp, "label": label_tmp})
        print(str(istep+1)+" / "+str(int(len(all_list)/step)))


def transfer_metric(model_name="phantom_model"):
    checkpoint_path = ".\\record\\" + model_name + "\\last_checkpoint.pth"
    tmp = tools.load_hyperparameters_metrics(file=checkpoint_path)
    hyper_parameters = tmp["hyper_parameters"]
    #  create a metric calculator
    metrics = tools.Metrics(loss_name=hyper_parameters["loss"])
    metrics.data = tmp["metrics"]

    train_score = metrics.data["mean_special_train_score"]
    test_score = metrics.data["mean_special_validate_score"]

    savemat(".\\record\\" + model_name + "\\metric.mat",
            {"train": train_score, "test": test_score})





if __name__ == "__main__":
    # test_sub_dataset(checkpoint_path=".\\record\\best_checkpoint.pth",
    #      dataset_dir=".\\data\\sub1",
    #      model_type="UNet3D",
    #      label_type="regression",
    #      device="cuda")
    pass

    # transfer_metric("phantom_model")
    # transfer_metric("fine_model_ssim_all")
    # transfer_metric("2D_ablation_all")

    # generate_mat_script(subdataset_name="V0-A0-PRF18K")
    # generate_mat_script(subdataset_name="V30-A0-PRF18K")
    # generate_mat_script(subdataset_name="V0-A90-PRF18K")
    # generate_mat_script(subdataset_name="V30-A90-PRF18K")
    # generate_mat_script(subdataset_name="p1-left-a0")
    # generate_mat_script(subdataset_name="p1-right-a0")
    # generate_mat_script(subdataset_name="p2-left-a0")
    # generate_mat_script(subdataset_name="p3-right-a0")
    # generate_mat_script(subdataset_name="p4-right-a0")

    # generate_mat_script_2D(subdataset_name="V0-A0-PRF18K")
    # generate_mat_script_2D(subdataset_name="V30-A0-PRF18K")
    # generate_mat_script_2D(subdataset_name="V0-A90-PRF18K")
    # generate_mat_script_2D(subdataset_name="V30-A90-PRF18K")