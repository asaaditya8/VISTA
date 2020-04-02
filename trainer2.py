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
screen = [416, 416]
device = torch.device('cuda')


def custom_loss(pprob, pbox, tprob, tbox):
    pprob = pprob.permute(0, 2, 3, 1)
    pbox = pbox.permute(0, 2, 3, 1)

    """
    Adjust prediction
    """
    cell_xy = torch.Tensor(np.arange(169).reshape(1, 13, 13, 1)).to(device)
    # because screen size is same for x and y
    pred_xy = (pbox[..., 0:2] + cell_xy) * screen[0]
    pred_wh = pbox[..., 2:4] * screen[0]
    pred_box_conf = pbox[..., 4]
    pred_class = pprob.permute(0, 3, 1, 2)

    """
    Adjust ground truth
    """
    tprob = tprob.reshape(-1, 13, 13)
    true_xy = (tbox[..., 0:2] + cell_xy) * screen[0]
    true_wh = tbox[..., 2:4] * screen[0]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = torch.max(pred_mins, true_mins)
    intersect_maxes = torch.max(pred_maxes, true_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.Tensor([0.]).to(device))
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    true_box_conf = tprob.float() * iou_scores
    true_box_conf = torch.max(true_box_conf, torch.Tensor([0.]).to(device))
    true_box_conf = torch.where(torch.isnan(true_box_conf), torch.Tensor([0.]).to(device), true_box_conf)

    """
    Determine the masks
    """
    # print('pred_box_conf', pred_box_conf.shape, pred_box_conf[0])
    # print('true_box_conf', true_box_conf.shape, true_box_conf[0])
    # print(torch.isnan(pbox))
    masked_box = pbox[..., 0:4] #* tprob.reshape(-1, 13, 13, 1).float()
    pxy = masked_box[..., 0:2]
    pwh = masked_box[..., 2:4]


    loss_c = F.nll_loss(pred_class, tprob)
    loss_r1 = (F.mse_loss(pxy, tbox[..., 0:2], reduce=False)).mean()
    loss_r2 = (F.mse_loss(pwh, tbox[..., 2:4], reduce=False)).mean()
    loss_p = F.l1_loss(pred_box_conf, true_box_conf)
    return loss_c, loss_r1 * 1e2, loss_r2 * 1e2, loss_p * 1e-3

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

                all_cur_loss = custom_loss(pprob, pbox, tprob, tbox)
                cur_loss = sum(all_cur_loss)
                # Optimization step
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step()

                # Accuracy Calculation
                loss.update(cur_loss.item())
                tqdm_batch.set_postfix(loss=loss.avg, acc=acc.avg, all_loss=str([a.item() for a in all_cur_loss]))

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
            cur_loss = custom_loss(pprob, pbox, tprob, tbox)

            # Accuracy Calculation
            loss.update(sum(cur_loss).item())


        print("Test Results" + " | " + "loss: " + str(loss.avg) + " - acc: " + str(acc.avg))

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.learning_rate * (self.learning_rate_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate


if __name__ == '__main__':
    model = HeadDetector().to(device)

    traindataset = SelfieDataset(bbox_csv, data_dir, mode='train')
    trainloader = DataLoader(traindataset, batch_size=4,
                            shuffle=True, num_workers=4)

    testdataset = SelfieDataset(bbox_csv, data_dir, mode='test')
    testloader = DataLoader(testdataset, batch_size=4,
                            shuffle=True, num_workers=4)

    model.unfreeze(False)

    for sample in trainloader:
        x, tprob, tbox = sample['image'], sample['true_prob'], sample['true_box']
        x, tprob, tbox = x.to(device), tprob.to(device), tbox.to(device)

        pprob, pbox = model(x)

        loss = custom_loss(pprob, pbox, tprob, tbox)
        print(loss)
        break