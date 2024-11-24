"""Reidentify visible views in a UI"""

import sys
from sys import stdout
from os.path import join, dirname, abspath
import os
from logging import Logger
from typing import Tuple

import torch.nn.functional as f
import numpy as np
from torch.utils.data import Dataset
import argparse
import torchvision.models
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import progressbar as bar
import torch
import torch.nn as nn

curpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(curpath)[0]
if rootpath not in sys.path:
    sys.path.append(rootpath)


class ImgDataSet(Dataset):
    """Build and load the view image dataset"""
    def __init__(self):
        npy_file = join(dirname(abspath(__file__)), "dataset_10.npy")
        self.data = np.load(npy_file, allow_pickle=True)
        self._classnum = np.max(self.data[:, 1]) + 1
        # shuffle the data by shuffling the indices
        self.index = [i for i in range(self.__len__())]
        np.random.shuffle(self.index)
        print(f"Dataset ready, {len(self.data)} items detected")

    def __getitem__(self, idx: int):
        idx = self.index[idx]
        i, label = self.data[idx]
        return i, label

    def __len__(self):
        return len(self.data)

    @property
    def class_num(self):
        return self._classnum


class ImgNet(nn.Module):
    def __init__(self, class_num: int):
        """A convolutional neural network based on ResNet

        Args:
            class_num: neural number of the output fc layer
        """
        super(ImgNet, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = torchvision.models.resnet18()
        fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(fc_in_features, class_num)
        self.target = [i for i in range(class_num)]

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        x = f.softmax(x, dim=1)
        return x


class ImgClassifier:
    def __init__(self,
                 logger: Logger,
                 lr_init: float = 0.001,
                 lr_decay: Tuple[int, float] = (10, 0.1),
                 epoch: int = 5,
                 batch_size: int = 32,
                 confidence_threshold: float = 0.95,
                 class_num: int = 10):
        """

        Args:
            lr_init (float): Initial value for the learning rate
            lr_decay: (int, float) - The first int indicates the
              epoch interval to decrease the learning rate. The
              second float is the decay ratio
            epoch (int): Training epoch
            batch_size (int): Training batch size
            confidence_threshold (float): The confidence to take the
              predicted label
        """
        self.logger = logger
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = epoch
        self.batch_size = batch_size
        self.class_num = class_num
        self.net = ImgNet(self.class_num).to(self.device)

        self.lr = lr_init
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.model_path = join(abspath(dirname(__file__)), "model_10.tar")
        self.confidence_threshold = confidence_threshold

    def deal_data(self, data):
        i, label = data
        i = np.array([a.unsqueeze_(0).numpy() for a in i])
        i = torch.from_numpy(i).float()
        label = label.long()
        i, label = i.to(self.device), label.to(self.device)
        return i, label

    def train(self):
        """training & validating"""

        self.dataset = ImgDataSet()
        s1 = int(0.8 * len(self.dataset))
        s2 = int(0.9 * len(self.dataset))  # 8:1:1
        index_list = list(range(len(self.dataset)))
        train_idx, valid_idx, test_idx =  index_list[:s1], index_list[s1:s2], index_list[s2:]

        trn_samp = sampler.SubsetRandomSampler(train_idx)
        val_samp = sampler.SubsetRandomSampler(valid_idx)
        trainloader = DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 sampler=trn_samp)
        validloader = DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 sampler=val_samp)
        testloader = DataLoader(torch.utils.data.Subset(self.dataset, test_idx), 
                                batch_size=self.batch_size)
        print(f"Train set: {len(train_idx)}, Val set: {len(valid_idx)}, Test set: {len(test_idx)}")

        for e in range(self.epoch):
            train_loss = []
            valid_loss = []
            self.net.train()
            pbar = bar.ProgressBar(
                widgets=[f"Training ({e + 1}/{self.epoch}) ",
                         bar.Percentage(), ' ', bar.Bar('=')],
                fd=stdout, maxval=len(trainloader))
            pbar.start()
            for batch_idx, data in enumerate(trainloader):
                pbar.update(batch_idx + 1)
                i, t = self.deal_data(data)
                self.optimizer.zero_grad()
                o = self.net(i)
                loss = f.nll_loss(torch.log(o), t)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            pbar.finish()
            self.net.eval()
            pbar2 = bar.ProgressBar(
                widgets=[f"Validating ({e + 1}/{self.epoch}) ",
                         bar.Percentage(), ' ', bar.Bar('=')],
                fd=stdout, maxval=len(validloader))
            pbar2.start()
            for batch_idx, data in enumerate(validloader):
                pbar2.update(batch_idx + 1)
                i, label = self.deal_data(data)
                o = self.net(i)
                loss = f.nll_loss(torch.log(o), label)
                valid_loss.append(loss.item())

            pbar2.finish()
            self.lr = self.lr_init * (self.lr_decay[1] ** int(e / self.lr_decay[0]))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            _loss_train = np.mean(train_loss)
            _loss_val = np.mean(valid_loss)
        
            print(f"Train Loss: {_loss_train:.2}, Valid Loss: {_loss_val:.2}")
            self.test(testloader)

        torch.save(self.net.state_dict(), self.model_path)

    def test(self, dataloader):
        """Test the model on a test dataset"""
        self.net.eval()
        sum_correct = 0
        sum_total = 0
        pbar = bar.ProgressBar(
            widgets=["Testing ", bar.Percentage(),
                     ' ', bar.Bar('=')],
            fd=stdout, maxval=len(dataloader))
        pbar.start()
        confuse_mat = np.zeros((self.dataset.class_num,
                                self.dataset.class_num))
        num_predict, num_predict_correct = 0, 0
        for batch_idx, data in enumerate(dataloader):
            u, t = self.deal_data(data)
            pbar.update(batch_idx + 1)
            o = self.net(u)
            output_vector = o.cpu().detach().numpy()
            output_labels = np.argmax(output_vector, axis=1)
            output_labels_confident = \
                [np.argmax(i) if max(i) > self.confidence_threshold
                 else -1 for i in output_vector]

            sum_total += t.size(0)
            tarray = t.cpu()
            num_predict += len([i for i in output_labels_confident if i > -1])
            num_predict_correct += \
                len([i for i in range(len(tarray)) if
                     output_labels_confident[i] == tarray[i] and
                     output_labels_confident[i] > -1])
            for o, t in zip(output_labels, tarray):
                confuse_mat[o, t] += 1

            a = len([i for i in range(len(tarray)) if
                     output_labels[i] == tarray[i]])
            sum_correct += a

        pbar.finish()
        print(f'Acc for {sum_total} samples: '
              f'{(sum_correct / sum_total * 100):.2f}%')
        print(f'Acc for {num_predict} samples (confidence={self.confidence_threshold}): '
              f'{(num_predict_correct / num_predict * 100):.2f}%')
        for i in range(self.class_num):
            if max(confuse_mat[i]) == 0:
                continue
            acc = confuse_mat[i][i] / sum(confuse_mat[i])
            print(f"Acc of {i}: {acc}")

    def load_model(self):
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.net.eval()

    def predict(self, images: np.array) -> list:
        """Predict views for a UI dataset

        Args:
            images (np.array): view images

        Returns:
            predicted view type labels
        """
        labels = []
        views_identified = 0
        for img in images:
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float().to(self.device)
            pre_vec = self.net(img)
            pre_vec_array = pre_vec.detach().cpu().numpy()
            max_value = np.max(pre_vec_array)
            if max_value > self.confidence_threshold:
                views_identified = views_identified + 1
                max_index = torch.Tensor.argmax(pre_vec)
                if self.device.type == "cpu":
                    pre_label = int(max_index.detach().numpy())
                else:
                    pre_label = int(max_index.cpu().numpy())
            else:
                # unidentified
                pre_label = -1

            labels.append(pre_label)

        info = f"Reidentified views: {views_identified}/{len(images)} " \
               f"({(views_identified / len(images) * 100.):.2f}%)"
        if self.logger is not None:
            self.logger.info(info)
        else:
            print(info)
            
        return labels


def parse_arg_reclass(input_args: list):
    parser = argparse.ArgumentParser(
        description="Reidenfity UI controls based on their image features")
    parser.add_argument("--input_path", "-i", help="input path")
    parser.add_argument("--lr", "-l", default=0.003, type=float,
                        help="training learning rate of the model")
    parser.add_argument("--decay", "-d", default='4,0.1', type=str,
                        help="training learning rate decay of the model, "
                             "format: decay_epoch,decay_rate")
    parser.add_argument("--batch_size", "-b", default=128, type=int,
                        help="training batch size of the model")
    parser.add_argument("--epoch", "-e", default=12, type=int,
                        help="training epoch of the model")
    parser.add_argument("--threshold", "-t", default=0.95, type=float,
                        help="prediction confidence of the model")
    parser.add_argument("--train", "-tt", action="store_true", default=False,
                        help="train model and overwrite the existing one (if have)")
    _args = parser.parse_args(input_args)
    return _args


if __name__ == '__main__':
    args = parse_arg_reclass(sys.argv[1:])

    try:
        lr_decay_e, lr_decay_r = args.decay.split(',')[0], args.decay.split(',')[1]
        lr_decay_e, lr_decay_r = int(lr_decay_e), float(lr_decay_r)
        ic = ImgClassifier(epoch=args.epoch, batch_size=args.batch_size,
                           lr_init=args.lr, lr_decay=(lr_decay_e, lr_decay_r),
                           confidence_threshold=args.threshold, logger=None)
        
        if args.train:
            ic.train()
    
        else:
            # output predictions for elements imgs
            ic.load_model()
            ic.predict(args.input_path)

    except ValueError:
        print("invalid decay for learning rate. example: 4,0.1")
        exit(1)