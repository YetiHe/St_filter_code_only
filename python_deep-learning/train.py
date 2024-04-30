import numpy as np
import torch
import tools
import math
import os
import json
import model
import data_set
from scipy.io import savemat
# import pytorch_ssim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

'''
Two models are scheduled to implement and train.
1: 3D-UNet
    1.1: regression (must-to-do)
    1.2: segmentation (optional; if regression fails, then pick up another assignment in my graduation thesis. :) )

2: 2D-UNet-LSTM (optional; I really want to try, if time permitting...)

data members:
    1. hyper_parameters (a dict-type, to record the self-defined hyperparameters)
    2. model (torch.nn.model)
    3. optimizer (torch.optim.adam)
    4. dataset (Dataset in torch.utils.data)
    5. dataloader (DataLoader in torch.utils.data)
    6. metrics (tools.Metric, a calculator and recorder for the metric)
'''


class Train:
    def __init__(self, learning_rate=1e-3, n_epochs=8, batch_size=12, dim=[64, 180], sequence_dim=200,
                 is_continue=False, change_continue_params={},
                 learning_rate_mode="normal", adam_weight_decay=1e-7, random_seed=643,
                 loss="SmoothL1Loss", sequence_step=100,
                 dataset_dir=".\\data", save_dir=tools.DEFAULT_SAVE_DIR, device="cuda",
                 continue_load_file=os.path.join(tools.DEFAULT_SAVE_DIR, tools.LAST_CHECKPOINT_NAME + ".pth"),
                 model_type="UNet3D", label_type="regression", is_tune=False):

        if not is_continue:
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                print("The device is not okay for gpu. It is automatically using cpu now...")
            #  create the self defined hyperparameters
            self.hyper_parameters = {"learning_rate": learning_rate,
                                     "n_epochs": n_epochs,
                                     "current_epoch": 1,
                                     "batch_size": batch_size,
                                     "dim": dim,
                                     "sequence_dim": sequence_dim,
                                     "is_continue": is_continue,
                                     "lr_mode": learning_rate_mode,
                                     "adam_weight_decay": adam_weight_decay,
                                     "random_seed": random_seed,
                                     "loss": loss,
                                     "label_type": label_type,
                                     "model_type": model_type,
                                     "device": device,
                                     "dataset_dir": dataset_dir,
                                     "save_dir": save_dir,
                                     "sequence_step": sequence_step}
            #  create the model
            self.model = self.choose_model(model_type, label_type)  # 选型
            self.model.to(torch.device(self.hyper_parameters["device"]))  # 使用对应设备
            #  create a metric calculator
            self.metrics = tools.Metrics(loss_name=self.hyper_parameters["loss"])
        else:
            if len(change_continue_params) == 0:
                #  create the model
                tmp = tools.load_hyperparameters_metrics(file=continue_load_file)

                self.hyper_parameters = tmp["hyper_parameters"]
                self.hyper_parameters["is_continue"] = True  # 如果是接着上次的继续，那么在start里需要注意载入checkpoint和model
                #  create a metric calculator
                self.metrics = tools.Metrics(loss_name=self.hyper_parameters["loss"])
                self.metrics.data = tmp["metrics"]

                self.model = self.choose_model(model_type=self.hyper_parameters["model_type"],
                                               label_type=self.hyper_parameters["label_type"])
                self.model, self.hyper_parameters = tools.load_model(
                    file=continue_load_file, model=self.model, hyper_para=self.hyper_parameters)  # 此时已经装好了模型以及设备，可以直接用

            else:
                #  create the model
                tmp = tools.load_hyperparameters_metrics(file=continue_load_file)

                self.hyper_parameters = tmp["hyper_parameters"]
                for key, value in change_continue_params.items():
                    self.hyper_parameters[key] = value  # 修改
                self.hyper_parameters["is_continue"] = True  # 如果是接着上次的继续，那么在start里需要注意载入checkpoint和model

                self.metrics = tools.Metrics(loss_name=self.hyper_parameters["loss"])
                if is_tune is False:
                    self.metrics.data = tmp["metrics"]

                self.model = self.choose_model(model_type=self.hyper_parameters["model_type"],
                                               label_type=self.hyper_parameters["label_type"])
                self.model, self.hyper_parameters = tools.load_model(
                    file=continue_load_file, model=self.model, hyper_para=self.hyper_parameters)

        tools.mkdir(self.hyper_parameters["save_dir"])
        #  create the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_parameters["learning_rate"],
                                          weight_decay=self.hyper_parameters["adam_weight_decay"])
        torch.manual_seed(self.hyper_parameters["random_seed"])
        np.random.seed(self.hyper_parameters["random_seed"])
        self.train_dataset = data_set.DataSet3D_all(self.hyper_parameters["dataset_dir"],
                                                    mode="train", label_type=self.hyper_parameters["label_type"],
                                                    dim=self.get_all_dim(),
                                                    sequence_step=self.hyper_parameters["sequence_step"])
        self.test_dataset = data_set.DataSet3D_all(self.hyper_parameters["dataset_dir"],
                                                   mode="test", label_type=self.hyper_parameters["label_type"],
                                                   dim=self.get_all_dim(),
                                                   sequence_step=self.hyper_parameters["sequence_step"])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.hyper_parameters["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(self.hyper_parameters["random_seed"]))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.hyper_parameters["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(self.hyper_parameters["random_seed"]))

    def choose_model(self, model_type, label_type):
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
        return md

    def get_all_dim(self):
        return [self.hyper_parameters["dim"][0], self.hyper_parameters["dim"][1], self.hyper_parameters["sequence_dim"]]

    def show_parameters(self):
        print("\n------------------Hyper-parameters---------------------")
        print("total epochs =", self.hyper_parameters["n_epochs"], "\ncurrent epoch =",
              self.hyper_parameters["current_epoch"] - 1)
        print("\ninitial learning rate =", self.hyper_parameters["learning_rate"],
              "\ncurrent learning rate =",
              self.__calculate_learning_rate__(), "\nlearning rate mode =",
              self.hyper_parameters["lr_mode"], "\nweight decay for adam optimizer =",
              self.hyper_parameters["adam_weight_decay"], "\nloss function: ",
              self.hyper_parameters["loss"])
        print("\nbatch size =", self.hyper_parameters["batch_size"], "\nstandard 2D dim =",
              self.hyper_parameters["dim"], "\nsequence (3rd dim) =",
              self.hyper_parameters["sequence_dim"],
              "\nsequence step =", self.hyper_parameters["sequence_step"])
        print("\nis continue:", self.hyper_parameters["is_continue"], "\nrandom seed:",
              self.hyper_parameters["random_seed"], "\ndataset dir:", self.hyper_parameters["dataset_dir"],
              "\nsave dir:", self.hyper_parameters["save_dir"])
        print("\ntraining set capacity =", len(self.train_dataloader),
              "\ntesting set capacity =", len(self.test_dataloader))
        print("-------------------------------------------------------\n")

    def train(self):
        if self.hyper_parameters["loss"] == "SmoothL1Loss":
            criterion = torch.nn.modules.loss.SmoothL1Loss(reduction='sum')
        elif self.hyper_parameters["loss"] == "MSE":
            criterion = torch.nn.modules.loss.MSELoss(reduction='sum')
        elif self.hyper_parameters["loss"] == "MAE":
            criterion = torch.nn.modules.loss.L1Loss(reduction='sum')
        elif self.hyper_parameters["loss"] == "SSIM":
            criterion = model.hyt_SSIM_loss()
        elif self.hyper_parameters["loss"] == "CrossEntropy":
            criterion = torch.nn.modules.loss.CrossEntropyLoss()
        elif self.hyper_parameters["loss"] == "BCEWithLogitsLoss":
            criterion = torch.nn.modules.loss.BCEWithLogitsLoss()
        elif self.hyper_parameters["loss"] == "BCELoss":
            criterion = torch.nn.modules.loss.BCELoss()
        elif self.hyper_parameters["loss"] == "self":
            criterion = model.hyt_SSIM_Huber_loss()
        else:  # self-defined loss function, probably dice score
            criterion = model.loss_demo()

        #  需要能支持自定义的子数据域训练
        # epochs
        for i in range(self.hyper_parameters["current_epoch"] - 1, self.hyper_parameters["n_epochs"]):
            tmp_metrics = self.__an_epoch__(criterion, mode="train")
            self.metrics.put_loss(tmp_metrics["train_loss"], is_train=1)
            self.metrics.put_score(tmp_metrics["train_score"], is_train=1)
            tmp_metrics = self.__an_epoch__(criterion, mode="test")
            self.metrics.put_loss(tmp_metrics["validate_loss"], is_train=0)
            self.metrics.put_score(tmp_metrics["validate_score"], is_train=0)
            self.__learning_rate_update__()
            self.hyper_parameters["current_epoch"] = self.hyper_parameters["current_epoch"] + 1
            tools.save_model(self.hyper_parameters["save_dir"],
                             "e" + str(i), self.model,
                             self.metrics.data, self.hyper_parameters)
            if self.metrics.is_the_best_for_now() is True:
                tools.save_checkpoint({"model_parameters": self.model.state_dict(),
                                       "metrics": self.metrics.data,
                                       "hyper_parameters": self.hyper_parameters},
                                      self.hyper_parameters["save_dir"], "best_checkpoint")
            if i + 1 > 1:
                if self.hyper_parameters["label_type"] != "regression":
                    self.metrics.draw_loss_line(save_dir=self.hyper_parameters["save_dir"], fig_handle_id=99, loss_name="BCE loss")
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=100, name_info="IOU", special_index=2,
                                                 is_confuse_matrix=True)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=101, name_info="dice", special_index=1,
                                                 is_confuse_matrix=True)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=102, name_info="accuracy", special_index=0,
                                                 is_confuse_matrix=True)
                else:
                    self.metrics.draw_loss_line(save_dir=self.hyper_parameters["save_dir"], fig_handle_id=99)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=100, name_info="SNR error ratio", special_index=0)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=101, name_info="CNR error ratio", special_index=1)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=102, name_info="SSIM", special_index=2)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=103, name_info="MSE loss", special_index=3)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=104, name_info="MAE loss", special_index=4)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=105, name_info="Huber loss", special_index=5)
                    self.metrics.draw_score_line(save_dir=self.hyper_parameters["save_dir"],
                                                 fig_handle_id=106, name_info="SSIM loss", special_index=6)

        #  the training step in every epoch. It might be called multiple times for the k-fold validation

    def __an_epoch__(self, criterion, mode="train"):
        print("epoch", self.hyper_parameters["current_epoch"], "/", self.hyper_parameters["n_epochs"], "is going through ...")
        tmp_metrics = {"train_loss": [],
                       "validate_loss": [],
                       "train_score": [],
                       "validate_score": []}

        if mode == "train":
            dataloader = self.train_dataloader
        else:
            dataloader = self.test_dataloader

        for i, (samples, regressions, masks) in enumerate(dataloader):
            if self.hyper_parameters["label_type"] != "regression":
                labels = masks  # 如果是分割任务，那取出来的回归regressions就用不上了，此时mask才是label
            else:
                labels = regressions
            output_size_info = labels.shape
            # 如果要改成十倍交叉验证的话（每个epoch都变一个验证集），此处最好是可以把判定条件写成函数来决定接下来这块究竟是train还是validate
            # 因为取来的samples本身已经被按照正确顺序堆叠了，所以不需要担心序列被破坏。
            if mode == "train":
                print("    batch", i + 1, "/", len(dataloader), "is training...")
                self.model.train()
                self.optimizer.zero_grad()
                if torch.cuda.is_available() and self.hyper_parameters["device"] == "cuda":
                    samples = samples.cuda()
                    labels = labels.cuda()

                if self.hyper_parameters["model_type"] != "UNet2D":
                    outputs = self.model(samples)
                else:
                    outputs = torch.zeros(size=samples.shape)
                    if torch.cuda.is_available() and self.hyper_parameters["device"] == "cuda":
                        outputs = outputs.cuda()
                    averaged_2D_loss = 0
                    for sub_id in range(0, samples.shape[4]):
                        tmp1 = self.model(samples[:,:,:,:,sub_id])
                        outputs[:,:,:,:,sub_id] = tmp1
                        sub_loss = criterion(tmp1, labels[:,:,:,:,sub_id])
                        sub_loss.backward()
                        averaged_2D_loss += sub_loss.item()
                    averaged_2D_loss /= samples.shape[4]

                # print(outputs.is_cuda, ", ", outputs.dtype, ", ", outputs.shape)
                if self.hyper_parameters["label_type"] != "regression":
                    labels = labels.view(-1)
                    outputs = outputs.view(-1)

                if self.hyper_parameters["model_type"] != "UNet2D":
                    loss = criterion(outputs, labels)
                    loss.backward()
                else:
                    loss = averaged_2D_loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)  # 防止梯度爆炸用的，2是个乘法因子
                self.optimizer.step()

                if self.hyper_parameters["model_type"] != "UNet2D":
                    if self.hyper_parameters["label_type"] != "regression" or self.hyper_parameters["loss"] == "SSIM" or self.hyper_parameters["loss"] == "self":
                        print("      loss=", loss.item())
                        tmp_metrics["train_loss"].append(loss.item())
                    else:
                        print("      loss=", loss.item() / (labels.view(-1)).shape[0])
                        tmp_metrics["train_loss"].append(loss.item() / (labels.view(-1)).shape[0])
                else:
                    print("      loss=", loss / (labels.view(-1)).shape[0])
                    tmp_metrics["train_loss"].append(loss / (labels.view(-1)).shape[0])

                if self.hyper_parameters["label_type"] != "regression":
                    if i + 2 == len(dataloader):
                        predict_tmp = tools.transform_binary_image(seq=outputs.cpu(), size5D=output_size_info,
                                                                   ti="train e" + str(self.hyper_parameters[
                                                                                         "current_epoch"]) + " 1 predict",
                                                                   save_dir=self.hyper_parameters[
                                                                                "save_dir"] + "\\train",
                                                                   select_batch=0)
                        label_tmp = tools.transform_binary_image(seq=outputs.cpu(),size5D=output_size_info,
                                                                 ti="train e" + str(self.hyper_parameters[
                                                                                       "current_epoch"]) + " 1 label",
                                                                 save_dir=self.hyper_parameters["save_dir"] + "\\train",
                                                                 select_batch=0)
                        savemat(
                            self.hyper_parameters["save_dir"] + "\\train\\" + "train-e" + str(self.hyper_parameters[
                                                                                                "current_epoch"]) + "-1.mat",
                            {"predict": predict_tmp, "label": label_tmp})

                        if outputs.shape[0] > 1:
                            predict_tmp = tools.transform_binary_image(seq=outputs.cpu(), size5D=output_size_info,
                                                                       ti="train e" + str(self.hyper_parameters[
                                                                                             "current_epoch"]) + " 2 predict",
                                                                       save_dir=self.hyper_parameters[
                                                                                    "save_dir"] + "\\train",
                                                                       select_batch=1)
                            label_tmp = tools.transform_binary_image(seq=outputs.cpu(),size5D=output_size_info,
                                                                     ti="train e" + str(self.hyper_parameters[
                                                                                           "current_epoch"]) + " 2 label",
                                                                     save_dir=self.hyper_parameters[
                                                                                  "save_dir"] + "\\train",
                                                                     select_batch=1)
                            savemat(
                                self.hyper_parameters["save_dir"] + "\\train\\" + "train-e" + str(
                                    self.hyper_parameters[
                                        "current_epoch"]) + "-2.mat",
                                {"predict": predict_tmp, "label": label_tmp})

                        if outputs.shape[0] > 2:
                            predict_tmp = tools.transform_binary_image(seq=outputs.cpu(), size5D=output_size_info,
                                                                       ti="train e" + str(self.hyper_parameters[
                                                                                             "current_epoch"]) + " 3 predict",
                                                                       save_dir=self.hyper_parameters[
                                                                                    "save_dir"] + "\\train",
                                                                       select_batch=2)
                            label_tmp = tools.transform_binary_image(seq=outputs.cpu(),size5D=output_size_info,
                                                                     ti="train e" + str(self.hyper_parameters[
                                                                                           "current_epoch"]) + " 3 label",
                                                                     save_dir=self.hyper_parameters[
                                                                                  "save_dir"] + "\\train",
                                                                     select_batch=2)
                            savemat(
                                self.hyper_parameters["save_dir"] + "\\train\\" + "train-e" + str(
                                    self.hyper_parameters[
                                        "current_epoch"]) + "-3.mat",
                                {"predict": predict_tmp, "label": label_tmp})

                    score_set = tools.calculate_confuse_matrix_score(outputs, labels)
                    print("      special score(accuracy; dice; IOU): ",
                          tools.calculate_confuse_based_score(score_set))
                else:
                    if i + 2 == len(dataloader):
                        predict_tmp = tools.transform_normal_image(data=outputs[0, :, :, :, :].cpu(),
                                                     ti="train e" + str(self.hyper_parameters[
                                                                            "current_epoch"]) + " 1 predict",
                                                     save_dir=self.hyper_parameters["save_dir"] + "\\train")
                        label_tmp = tools.transform_normal_image(data=labels[0, :, :, :, :].cpu(),
                                                     ti="train e" + str(self.hyper_parameters[
                                                                            "current_epoch"]) + " 1 label",
                                                     save_dir=self.hyper_parameters["save_dir"] + "\\train")
                        savemat(self.hyper_parameters["save_dir"] + "\\train\\"+"train-e" + str(self.hyper_parameters[
                                                                            "current_epoch"]) + "-1.mat",
                                {"predict":predict_tmp, "label":label_tmp})

                        if outputs.shape[0] > 1:
                            predict_tmp = tools.transform_normal_image(data=outputs[1, :, :, :, :].cpu(),
                                                         ti="train e" + str(self.hyper_parameters[
                                                                                "current_epoch"]) + " 2 predict",
                                                         save_dir=self.hyper_parameters["save_dir"] + "\\train")
                            label_tmp = tools.transform_normal_image(data=labels[1, :, :, :, :].cpu(),
                                                         ti="train e" + str(self.hyper_parameters[
                                                                                "current_epoch"]) + " 2 label",
                                                         save_dir=self.hyper_parameters["save_dir"] + "\\train")
                            savemat(
                                self.hyper_parameters["save_dir"] + "\\train\\" + "train-e" + str(self.hyper_parameters[
                                                                                                      "current_epoch"]) + "-2.mat",
                                {"predict": predict_tmp, "label": label_tmp})

                        if outputs.shape[0] > 2:
                            predict_tmp = tools.transform_normal_image(data=outputs[2, :, :, :, :].cpu(),
                                                         ti="train e" + str(self.hyper_parameters[
                                                                            "current_epoch"]) + " 3 predict",
                                                     save_dir=self.hyper_parameters["save_dir"] + "\\train")
                            label_tmp = tools.transform_normal_image(data=labels[2, :, :, :, :].cpu(),
                                                         ti="train e" + str(self.hyper_parameters[
                                                                                "current_epoch"]) + " 3 label",
                                                         save_dir=self.hyper_parameters["save_dir"] + "\\train")
                            savemat(
                                self.hyper_parameters["save_dir"] + "\\train\\" + "train-e" + str(self.hyper_parameters[
                                                                                                      "current_epoch"]) + "-3.mat",
                                {"predict": predict_tmp, "label": label_tmp})

                    score_set = tools.calculate_US_regression_ratio(outputs, labels, masks)
                    print("      special score(SNR error ratio; CNR error ratio; SSIM; MSE; MAE; Huber; SSIM): ", score_set)
                tmp_metrics["train_score"].append(score_set)
                # arr.append(loss.item())
                # tmp_metrics["train_loss"] = arr
            else:
                # test
                print("   testing:")
                print("    batch", i + 1, "/", len(dataloader), "is testing...")
                self.model.eval()
                with torch.no_grad():
                    if torch.cuda.is_available() and self.hyper_parameters["device"] == "cuda":
                        samples = samples.cuda()
                        labels = labels.cuda()

                    if self.hyper_parameters["model_type"] != "UNet2D":
                        outputs = self.model(samples)
                    else:
                        outputs = torch.zeros(size=samples.shape)
                        if torch.cuda.is_available() and self.hyper_parameters["device"] == "cuda":
                            outputs = outputs.cuda()
                        averaged_2D_loss = 0
                        for sub_id in range(0, samples.shape[4]):
                            tmp1 = self.model(samples[:, :, :, :, sub_id])
                            outputs[:, :, :, :, sub_id] = tmp1
                            sub_loss = criterion(tmp1, labels[:, :, :, :, sub_id])
                            averaged_2D_loss += sub_loss.item()
                        averaged_2D_loss /= samples.shape[4]

                    output_size_info = outputs.shape

                    if self.hyper_parameters["label_type"] != "regression":
                        labels = labels.view(-1)
                        outputs = outputs.view(-1)

                    if self.hyper_parameters["model_type"] != "UNet2D":
                        loss = criterion(outputs, labels)
                    else:
                        loss = averaged_2D_loss

                    if self.hyper_parameters["model_type"] != "UNet2D":
                        if self.hyper_parameters["label_type"] != "regression" or self.hyper_parameters["loss"] == "SSIM" or self.hyper_parameters["loss"] == "self":
                            print("      loss=", loss.item())
                            tmp_metrics["validate_loss"].append(loss.item())
                        else:
                            print("      loss=", loss.item() / (labels.view(-1)).shape[0])
                            tmp_metrics["validate_loss"].append(loss.item() / (labels.view(-1)).shape[0])
                    else:
                        print("      loss=", loss / (labels.view(-1)).shape[0])
                        tmp_metrics["validate_loss"].append(loss / (labels.view(-1)).shape[0])

                    if self.hyper_parameters["label_type"] != "regression":
                        if i + 2 == len(dataloader):
                            predict_tmp = tools.transform_binary_image(seq=outputs.cpu(),size5D=output_size_info,
                                                         ti="test e" + str(self.hyper_parameters[
                                                                                "current_epoch"]) + " 1 predict",
                                                         save_dir=self.hyper_parameters["save_dir"] + "\\test",
                                                                       select_batch=0)
                            label_tmp = tools.transform_binary_image(seq=outputs.cpu(),size5D=output_size_info,
                                                         ti="test e" + str(self.hyper_parameters[
                                                                               "current_epoch"]) + " 1 label",
                                                         save_dir=self.hyper_parameters["save_dir"] + "\\test",
                                                                     select_batch=0)
                            savemat(
                                self.hyper_parameters["save_dir"] + "\\test\\" + "test-e" + str(self.hyper_parameters[
                                                                                                      "current_epoch"]) + "-1.mat",
                                {"predict": predict_tmp, "label": label_tmp})

                            if outputs.shape[0] > 1:
                                predict_tmp = tools.transform_binary_image(seq=outputs.cpu(), size5D=output_size_info,
                                                                           ti="test e" + str(self.hyper_parameters[
                                                                                                 "current_epoch"]) + " 2 predict",
                                                                           save_dir=self.hyper_parameters[
                                                                                        "save_dir"] + "\\test",
                                                                           select_batch=1)
                                label_tmp = tools.transform_binary_image(seq=outputs.cpu(),size5D=output_size_info,
                                                                         ti="test e" + str(self.hyper_parameters[
                                                                                               "current_epoch"]) + " 2 label",
                                                                         save_dir=self.hyper_parameters[
                                                                                      "save_dir"] + "\\test",
                                                                         select_batch=1)
                                savemat(
                                    self.hyper_parameters["save_dir"] + "\\test\\" + "test-e" + str(
                                        self.hyper_parameters[
                                            "current_epoch"]) + "-2.mat",
                                    {"predict": predict_tmp, "label": label_tmp})

                            if outputs.shape[0] > 2:
                                predict_tmp = tools.transform_binary_image(seq=outputs.cpu(), size5D=output_size_info,
                                                                           ti="test e" + str(self.hyper_parameters[
                                                                                                 "current_epoch"]) + " 3 predict",
                                                                           save_dir=self.hyper_parameters[
                                                                                        "save_dir"] + "\\test",
                                                                           select_batch=2)
                                label_tmp = tools.transform_binary_image(seq=outputs.cpu(),size5D=output_size_info,
                                                                         ti="test e" + str(self.hyper_parameters[
                                                                                               "current_epoch"]) + " 3 label",
                                                                         save_dir=self.hyper_parameters[
                                                                                      "save_dir"] + "\\test",
                                                                         select_batch=2)
                                savemat(
                                    self.hyper_parameters["save_dir"] + "\\test\\" + "test-e" + str(
                                        self.hyper_parameters[
                                            "current_epoch"]) + "-3.mat",
                                    {"predict": predict_tmp, "label": label_tmp})

                        score_set = tools.calculate_confuse_matrix_score(outputs, labels)
                        print("      special score(accuracy; dice; IOU):",
                              tools.calculate_confuse_based_score(score_set))
                    else:
                        if i + 2 == len(dataloader):
                            predict_tmp = tools.transform_normal_image(data=outputs[0, :, :, :, :].cpu(),
                                                         ti="test e" + str(self.hyper_parameters[
                                                                                "current_epoch"]) + " 1 predict",
                                                         save_dir=self.hyper_parameters["save_dir"] + "\\test")
                            label_tmp = tools.transform_normal_image(data=labels[0, :, :, :, :].cpu(),
                                                         ti="test e" + str(self.hyper_parameters[
                                                                               "current_epoch"]) + " 1 label",
                                                         save_dir=self.hyper_parameters["save_dir"] + "\\test")
                            savemat(
                                self.hyper_parameters["save_dir"] + "\\test\\" + "test-e" + str(self.hyper_parameters[
                                                                                                      "current_epoch"]) + "-1.mat",
                                {"predict": predict_tmp, "label": label_tmp})

                            if outputs.shape[0] > 1:
                                predict_tmp = tools.transform_normal_image(data=outputs[1, :, :, :, :].cpu(),
                                                             ti="test e" + str(self.hyper_parameters[
                                                                                    "current_epoch"]) + " 2 predict",
                                                             save_dir=self.hyper_parameters["save_dir"] + "\\test")
                                label_tmp = tools.transform_normal_image(data=labels[1, :, :, :, :].cpu(),
                                                             ti="test e" + str(self.hyper_parameters[
                                                                                   "current_epoch"]) + " 2 label",
                                                             save_dir=self.hyper_parameters["save_dir"] + "\\test")
                                savemat(
                                    self.hyper_parameters["save_dir"] + "\\test\\" + "test-e" + str(
                                        self.hyper_parameters[
                                            "current_epoch"]) + "-2.mat",
                                    {"predict": predict_tmp, "label": label_tmp})

                            if outputs.shape[0] > 2:
                                predict_tmp = tools.transform_normal_image(data=outputs[2, :, :, :, :].cpu(),
                                                             ti="test e" + str(self.hyper_parameters[
                                                                                   "current_epoch"]) + " 3 predict",
                                                             save_dir=self.hyper_parameters["save_dir"] + "\\test")
                                label_tmp = tools.transform_normal_image(data=labels[2, :, :, :, :].cpu(),
                                                             ti="test e" + str(self.hyper_parameters[
                                                                                    "current_epoch"]) + " 3 label",
                                                             save_dir=self.hyper_parameters["save_dir"] + "\\test")
                                savemat(
                                    self.hyper_parameters["save_dir"] + "\\test\\" + "test-e" + str(
                                        self.hyper_parameters[
                                            "current_epoch"]) + "-3.mat",
                                    {"predict": predict_tmp, "label": label_tmp})

                        score_set = tools.calculate_US_regression_ratio(outputs, labels, masks)
                        print("      special score(SNR error ratio; CNR error ratio; SSIM; MSE; MAE; Huber; SSIM): ", score_set)
                    tmp_metrics["validate_score"].append(score_set)
        torch.cuda.empty_cache()
        return tmp_metrics

    def __calculate_learning_rate__(self):
        epoch = self.hyper_parameters["current_epoch"]
        if self.hyper_parameters["lr_mode"] == "cosine":
            lr_min = 0
            lr = math.fabs(lr_min + (1 + math.cos(1 * epoch * math.pi /
                                                  self.hyper_parameters["n_epochs"])) *
                           (self.hyper_parameters["learning_rate"] - lr_min) / 2.)
        else:
            lr = self.hyper_parameters["learning_rate"] * (0.5 ** (epoch // 10))  # 普通的衰减 每10个epoch用一次
        return lr

    def __learning_rate_update__(self):  # 调整学习速率
        lr = self.__calculate_learning_rate__()
        print("learning rate is %f " % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr  # 更新亚当优化器的学习速率


if __name__ == "__main__":

    # trainer1 = Train(dim=[64, 176], sequence_dim=200, sequence_step=200, learning_rate_mode="cosine",
    #                  batch_size=6, n_epochs=8, label_type="regression", loss="SmoothL1Loss",
    #                  model_type="UNet2D", learning_rate=1e-3,
    #                  save_dir=".\\record\\2D_ablation_all",
    #                  dataset_dir=".\\phantom_data", is_continue=False,
    #                  continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={})
    # trainer1.show_parameters()
    # trainer1.train()
    # trainer1.show_parameters()
    # trainer4 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data2",
    #                                          "save_dir": ".\\record\\fine_model_ssim_all",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-3,
    #                                          "sequence_step": 100,
    #                                          "n_epochs": 32,
    #                                          "lr_mode": "cosine",
    #                                          "loss": "SSIM"
    #                                          })
    # trainer4.show_parameters()
    # trainer4.train()
    # trainer4.show_parameters()

    trainer4 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\2D_ablation_all\\last_checkpoint.pth",
                     change_continue_params={"dataset_dir": ".\\fine_data2",
                                             "save_dir": ".\\record\\fine_model_ssim_all_2D",
                                             "current_epoch": 1,
                                             "learning_rate": 1e-3,
                                             "sequence_step": 100,
                                             "n_epochs": 32,
                                             "lr_mode": "cosine",
                                             "loss": "SSIM"
                                             })
    trainer4.show_parameters()
    trainer4.train()
    trainer4.show_parameters()

    # trainer4 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data",
    #                                          "save_dir": ".\\record\\fine_model_self_design",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-3,
    #                                          "sequence_step": 100,
    #                                          "n_epochs": 8,
    #                                          "lr_mode": "cosine",
    #                                          "loss": "self"
    #                                          })
    # trainer4.show_parameters()
    # trainer4.train()
    # trainer4.show_parameters()

    # trainer4 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\2D_ablation_all\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data",
    #                                          "save_dir": ".\\record\\fine_model_2D_try_fine1",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 5e-4,
    #                                          "sequence_step": 200,
    #                                          "loss": "MAE"
    #                                          })
    # trainer4.show_parameters()
    # trainer4.train()
    # trainer4.show_parameters()
    # trainer1 = Train(dim=[64, 176], sequence_dim=200, sequence_step=100, learning_rate_mode="cosine",
    #                  batch_size=6, n_epochs=8, label_type="regression", loss="MAE",
    #                  model_type="UNet3D", learning_rate=1e-3,
    #                  save_dir=".\\record\\try_fine_data1_sequential_spatial_atten",
    #                  dataset_dir=".\\fine_data", is_continue=True,
    #                  continue_load_file=".\\record\\try_fine_data1_sequential_spatial_atten\\last_checkpoint.pth",
    #                  change_continue_params={})
    # trainer1.show_parameters()
    # trainer1.train()
    # trainer1.show_parameters()
    # trainer1 = Train(dim=[64, 176], sequence_dim=200, sequence_step=100, learning_rate_mode="cosine",
    #                  batch_size=6, n_epochs=8, label_type="regression", loss="MAE",
    #                  model_type="UNet3D", learning_rate=1e-3, save_dir=".\\record\\try_fine_data1_sequential_spatial_atten",
    #                  dataset_dir=".\\fine_data", is_continue=False,
    #                  change_continue_params={})
    # trainer1.show_parameters()
    # trainer1.train()
    # trainer1.show_parameters()
#
    # trainer1 = Train(dim=[64, 176], sequence_dim=200, sequence_step=100, learning_rate_mode="cosine",
    #                  batch_size=6, n_epochs=8, label_type="segmentation", loss="BCELoss",
    #                  model_type="UNet3D", learning_rate=1e-3, save_dir=".\\record\\seg_try_phantom_data1",
    #                  dataset_dir=".\\data3", is_continue=False,
    #                  change_continue_params={})
    # trainer1.show_parameters()
    # trainer1.train()
    # trainer1.show_parameters()


#
    # trainer1 = Train(dim=[64, 176], sequence_dim=200, sequence_step=100, learning_rate_mode="cosine",
    #                  batch_size=6, n_epochs=8, label_type="regression", loss="SmoothL1Loss",
    #                  model_type="UNet3D", learning_rate=1e-3, save_dir=".\\record\\try-direct_fine_data2345",
    #                  dataset_dir=".\\fine_data2", is_continue=False,
    #                  change_continue_params={})
    # trainer1.show_parameters()
    # trainer1.train()
    # trainer1.show_parameters()

    # trainer1 = Train(dim=[64, 176], sequence_dim=200, sequence_step=100, learning_rate_mode="cosine",
    #                  batch_size=6, n_epochs=8, label_type="regression", loss="SmoothL1Loss",
    #                  model_type="UNet3D", learning_rate=1e-3, save_dir=".\\record\\phantom_model",
    #                  dataset_dir=".\\phantom_data", is_continue=False,
    #                  change_continue_params={})
    # trainer1.show_parameters()
    # trainer1.train()
    # trainer1.show_parameters()
    # trainer1 = Train(continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth", is_continue=True,
    #                  change_continue_params={})
    # trainer1.show_parameters()
    # trainer1.train()
    # trainer1.show_parameters()

    # trainer4 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\pha_data_noise",
    #                                          "save_dir": ".\\record\\pha_noise_model_MAE",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-3,
    #                                          "sequence_step": 100,
    #                                          "loss": "MAE"
    #                                          })
    # trainer4.show_parameters()
    # trainer4.train()
    # trainer4.show_parameters()

    # trainer4 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data_filter",
    #                                          "save_dir": ".\\record\\fine_modelMAE_filter_dim320",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-3,
    #                                          "sequence_step": 100,
    #                                          "sequence_dim": 320,
    #                                          "batch_size": 4,
    #                                          "loss": "MAE"
    #                                          })
    # trainer4.show_parameters()
    # trainer4.train()
    # trainer4.show_parameters()
#
    # trainer4 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data_filter",
    #                                          "save_dir": ".\\record\\fine_model_SLL_filter_dim320",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-3,
    #                                          "sequence_step": 100,
    #                                          "sequence_dim": 320,
    #                                          "batch_size": 4,
    #                                          "loss": "SmoothL1Loss"
    #                                          })
    # trainer4.show_parameters()
    # trainer4.train()
    # trainer4.show_parameters()

#
#
    # trainer7 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data",
    #                                          "save_dir": ".\\record\\fine_model_MAE_smaller_lr",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-4,
    #                                          "sequence_step": 100,
    #                                          "loss": "MAE"
    #                                          })
    # trainer7.show_parameters()
    # trainer7.train()
    # trainer7.show_parameters()


    # trainer2 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data",
    #                                          "save_dir": ".\\record\\fine_model",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-3,
    #                                          "sequence_step": 100
    #                                          })
    # trainer2.show_parameters()
    # trainer2.train()
    # trainer2.show_parameters()
#
#
    # trainer3 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data",
    #                                          "save_dir": ".\\record\\fine_model_smaller_step",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-3,
    #                                          "sequence_step": 50,
    #                                          "loss": "MAE"
    #                                          })
    # trainer3.show_parameters()
    # trainer3.train()
    # trainer3.show_parameters()
#
#
    # trainer6 = Train(is_continue=True, is_tune=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data",
    #                                          "save_dir": ".\\record\\fine_model_MAE_more_data_smaller_lr_normal_adjust",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-3,
    #                                          "sequence_step": 50,
    #                                          "lr_mode": "normal",
    #                                          "loss": "MAE"
    #                                          })
    # trainer6.show_parameters()
    # trainer6.train()
    # trainer6.show_parameters()

    # trainer3 = Train(is_continue=True, continue_load_file=".\\record\\phantom_model\\last_checkpoint.pth",
    #                  change_continue_params={"dataset_dir": ".\\fine_data",
    #                                          "save_dir": ".\\record\\fine_model_smaller_lr",
    #                                          "current_epoch": 1,
    #                                          "learning_rate": 1e-4
    #                                          })
    # trainer3.show_parameters()
    # trainer3.train()
    # trainer3.show_parameters()

    '''
    Train的三种常用用法：
    1. 正常使用，如上，可以直接调用Train, 所有参数都有默认值(默认不划分专门的验证集，直接拿测试集做验证)

    2. 载入上一次训练到一半被停掉的模型接着继续训练（常见场景：训练了一晚上没训练完，早上别人要用服务器，于是你需要暂时中断训练的时候），则初始化时的
        is_continue=True，其余参数会自动无效化，程序自动载入上一次的模型的所有参数和超参并接着上一次的epoch继续训练

    3. 载入某一次的模型训练，但是想要改变超参（常见场景：拿已经用仿体数据集训练好了初值的模型，去做体内数据集的微调），则初始化时：
        is_continue=True；
        continue_load_file="你要载入的模型的.pth文件"（文件已含之前训练该模型时定义的超参）；
        change_continue_params={这是一个字典，其键值对需要符合Train类中self.hyper_parameters字典中定义的键值对，Train会自动替换你给的新值到对应的超参键上}

    ps. 如果改变了模型结构，需要自行扒开.pth文件去做参数减枝
    '''

