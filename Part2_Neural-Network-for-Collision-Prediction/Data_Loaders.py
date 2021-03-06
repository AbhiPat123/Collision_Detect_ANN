import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.utils import resample


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        # Data Balance Trick 1: removing redundant rows
        data_uniq = np.unique(self.data, axis=0)

        # Data Balance Trick 2: upsample class for collision
        # separate both classes
        data_uniq_lbl0 = data_uniq[np.where(data_uniq[:,-1]==0)]
        data_uniq_lbl1 = data_uniq[np.where(data_uniq[:,-1]==1)]

        # up-sample the collision class
        data_uniq_lbl1_up_sampled = resample(data_uniq_lbl1, replace=True, n_samples=len(data_uniq_lbl0))

        # recombine data
        balanced_data = np.row_stack((data_uniq_lbl0, data_uniq_lbl1_up_sampled))

        # shuffle data for better learning and update self.data
        np.random.shuffle(balanced_data)
        self.data = balanced_data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.

        # get the input and label values as float32 dtype
        x = torch.tensor(self.normalized_data[idx][:-1], dtype=torch.float32)
        y = torch.tensor(self.normalized_data[idx][-1], dtype=torch.float32)

        # create a return dictionary
        ret_dict = {
            'input': x,
            'label': y
        }
        return ret_dict

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        # get the length of the entire dataset
        dataset_len = self.nav_dataset.__len__()

        # set training and testing data size
        tr_data_size = int(0.8 * dataset_len)
        ts_data_size = dataset_len - tr_data_size

        # use torch's random_split to randomly split the dataset
        tr_dataset, ts_dataset = data.random_split(self.nav_dataset, [tr_data_size, ts_data_size])

        # create two data loaders one each for train and test
        self.train_loader = data.DataLoader(dataset=tr_dataset, batch_size=batch_size)
        self.test_loader = data.DataLoader(dataset=ts_dataset, batch_size=batch_size)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
