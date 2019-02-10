import torch
from torch.utils.data import DataLoader

from model import HeadDetector
from trainer2 import Train
from data import SelfieDataset

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

    model.unfreeze(False)
    trainer = Train(model, trainloader, testloader, 1, 0.00001, 0.98, 1)
    trainer.train()
    # trainer.model.unfreeze(True)
    # trainer.num_epochs = 1
    # trainer.train()

    torch.save(model.state_dict(), '/home/aaditya/PycharmProjects/VISTA/yolo_selfie_8.pth')