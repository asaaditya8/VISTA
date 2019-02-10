import torch
from torch.utils.data import DataLoader
import numpy as np
from bbox import process_csv, get_gt
from preprocess import prep_image


data_dir = '/home/aaditya/PycharmProjects/VISTA/Data/image_data/'
bbox_csv = '/home/aaditya/PycharmProjects/VISTA/Data/bbox_train.csv'


class SelfieDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, inp_dim=416, grid_size=13, test_fraction=0.2, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dict = process_csv(csv_file, test_fraction, mode)
        self.root_dir = root_dir
        self.inp_dim = inp_dim
        self.grid_size = grid_size

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        img_filename = list(self.data_dict.keys())[idx]
        img = prep_image(self.root_dir + img_filename, self.inp_dim)

        item_dict = self.data_dict[img_filename]
        tprob, tbox = get_gt(item_dict['img_dim'], self.grid_size, np.array(item_dict['bbox'], dtype='float32'))

        sample = {'image': img, 'true_prob': tprob, 'true_box': tbox}

        return sample

if __name__ == '__main__':
    dataset = SelfieDataset(bbox_csv, data_dir)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    for sample in dataloader:
        print(
            sample['image'].size(),
            sample['true_prob'].size(),
            sample['true_box'].size()
        )
        break