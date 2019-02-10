import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from model import HeadDetector
from itertools import chain
import numpy as np

from data import SelfieDataset

data_dir = '/home/aaditya/PycharmProjects/VISTA/Data/image_data/'
bbox_csv = '/home/aaditya/PycharmProjects/VISTA/Data/bbox_train.csv'


class AverageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Train:
    def __init__(self, model, trainloader, valloader, num_epochs, learning_rate, learning_rate_decay, test_every):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.test_every = test_every
        self.start_epoch = 0
        self.lamda = 0.06

        # Loss function and Optimizer
        self.loss_c = nn.NLLLoss(reduce=False)
        self.loss_r = nn.L1Loss(reduce=False)
        self.optimizer = torch.optim.Adam(chain(model.common.parameters(), model.classifier.parameters(), model.regressor.parameters()),
                                          lr=self.learning_rate, weight_decay=1e-6)

    def train(self):
        for cur_epoch in range(self.start_epoch, self.num_epochs):

            # Meters for tracking the average values
            loss = AverageTracker()
            acc = AverageTracker()

            # Initialize tqdm
            tqdm_batch = tqdm(self.trainloader, desc="Epoch-" + str(cur_epoch) + "-")

            # Learning rate adjustment
            self.adjust_learning_rate(self.optimizer, cur_epoch)

            # Set the model to be in training mode (for dropout and batchnorm)
            self.model.train()

            for sample in tqdm_batch:
                data, tprob, tbox = sample['image'], sample['true_prob'], sample['true_box']
                data, tprob, tbox = data.to(device), tprob.to(device), tbox.to(device)

                # Forward pass
                pprob, pbox = self.model(data)
                pprob = pprob.permute(0, 2, 3, 1)
                pbox = pbox.permute(0, 2, 3, 1)

                # cur_loss = self.mse_loss(pprob, tprob)

                # acc.update(1 - cur_loss.data.cpu().numpy().reshape(1)[0])

                cur_loss = (self.loss_r(pbox, tbox) * tprob.float()).sum()

                pprob = pprob.permute(0, 3, 1, 2)
                tprob = tprob.view(-1, 13, 13)
                cur_loss += self.lamda * self.loss_c(pprob, tprob).sum()

                cur_loss += (self.loss_c(pprob, tprob) * tprob.float()).sum()

                # Optimization step
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step()

                # Accuracy Calculation
                loss.update(cur_loss.data.cpu().numpy().reshape(1)[0])
                tqdm_batch.set_postfix(loss=loss.avg, acc=acc.avg)

            tqdm_batch.close()
            # Print in console
            print("Epoch-" + str(cur_epoch) + " | " + "loss: " + str(loss.avg) + "acc: " + str(acc.avg))

            # Evaluate on Validation Set
            if cur_epoch % self.test_every == 0 and self.valloader:
                self.test(self.valloader)

    def test(self, testloader):
        loss = AverageTracker()
        acc = AverageTracker()
        self.model.eval()

        for sample in testloader:
            data, tprob, tbox = sample['image'], sample['true_prob'], sample['true_box']
            data, tprob, tbox = data.to(device), tprob.to(device), tbox.to(device)

            # Forward pass
            pprob, pbox = self.model(data)
            pprob = pprob.permute(0, 2, 3, 1)
            pbox = pbox.permute(0, 2, 3, 1)

            cur_loss = self.mse_loss(pprob, tprob)
            acc.update(1 - cur_loss.data.cpu().numpy().reshape(1)[0])

            cur_loss += (self.loss_r(pbox, tbox) * tprob.float()).sum()

            pprob = pprob.permute(0, 3, 1, 2)
            tprob = tprob.view(-1, 13, 13)
            cur_loss += self.lamda * self.loss_c(pprob, tprob).sum()
            cur_loss += (self.loss_c(pprob, tprob) * tprob.float()).sum()

            # Accuracy Calculation
            loss.update(cur_loss.data.cpu().numpy().reshape(1)[0])


        print("Test Results" + " | " + "loss: " + str(loss.avg) + " - acc: " + str(acc.avg))

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.learning_rate * (self.learning_rate_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    def mse_loss(self, output, target):
        # output shape = batch_size x 13 x 13 x 2
        output = output.argmax(-1).sum(2).sum(1)
        target = target.view(-1, 13, 13).sum(2).sum(1)
        loss = F.mse_loss(output.float(), target.float()).sqrt()
        return loss


if __name__ == '__main__':
    device = torch.device('cuda')
    model = HeadDetector().to(device)

    traindataset = SelfieDataset(bbox_csv, data_dir, mode='train')
    trainloader = DataLoader(traindataset, batch_size=4,
                            shuffle=True, num_workers=4)

    testdataset = SelfieDataset(bbox_csv, data_dir, mode='test')
    testloader = DataLoader(testdataset, batch_size=4,
                            shuffle=True, num_workers=4)

    model.unfreeze(False)
    trainer = Train(model, trainloader, testloader, 1, 0.001, 0.98, 1)