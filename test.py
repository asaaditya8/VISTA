import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt, patches
from model import HeadDetector
import numpy as np

from data import SelfieDataset
from bbox import transform_coord

data_dir = '/home/aaditya/PycharmProjects/VISTA/Data/image_data/'
bbox_csv = '/home/aaditya/PycharmProjects/VISTA/Data/bbox_train.csv'


if __name__ == '__main__':

    device = torch.device('cuda')
    model = HeadDetector().to(device)

    traindataset = SelfieDataset(bbox_csv, data_dir, mode='train')
    trainloader = DataLoader(traindataset, batch_size=4,
                             shuffle=True, num_workers=4)

    testdataset = SelfieDataset(bbox_csv, data_dir, mode='test')
    testloader = DataLoader(testdataset, batch_size=4,
                            shuffle=True, num_workers=4)

    state_dict = torch.load('/home/aaditya/PycharmProjects/VISTA/yolo_selfie_8.pth')
    model.load_state_dict(state_dict)
    model.eval()
    for sample in testloader:
        x, tprob, tbox = sample['image'], sample['true_prob'], sample['true_box']
        x, tprob, tbox = x.to(device), tprob.to(device), tbox.to(device)
        pprob, pbox = model(x)
        break

    x = x.permute(0, 2, 3, 1).data.cpu().numpy()
    pprob = pprob.permute(0, 2, 3, 1).data.cpu().numpy()
    pbox = pbox.permute(0, 2, 3, 1).data.cpu().numpy()
    pprob = pprob.argmax(-1)
    print('pprob shape: ', pprob.shape)
    pprob = pprob.reshape(-1, 169)
    print('pprob', pprob[0])
    pbox = pbox.reshape(-1, 169, 5)
    cells = np.where(pprob[0] > 0.1)[0]
    print('cells', cells)
    print('confidence ', pbox[0][..., 4])

    ax = plt.gca()
    ax.imshow(np.clip(x[0]*255, 0, 255).astype('uint8'))
    for n in cells:
        c1, c2, c3 = transform_coord(pbox[0][n][:4], n)
        print(c1, c2, c3)
        rec = patches.Rectangle(c1, c2+50, c3+50, edgecolor='r', fill=False, linewidth=2)
        ax.add_patch(rec)

    tprob = tprob.data.cpu().numpy()
    tbox = tbox.data.cpu().numpy()
    tprob = tprob.reshape(-1, 169)
    print('tprob', tprob[0])
    cells = np.where(tprob[0] > 0.03)[0]
    tbox = tbox.reshape(-1, 169, 4)

    for n in cells:
        c1, c2, c3 = transform_coord(tbox[0][n], n)
        print(int(c2), int(c3))
        rec = patches.Rectangle(c1, c2, c3, edgecolor='g', fill=False, linewidth=2)
        ax.add_patch(rec)

    plt.show()