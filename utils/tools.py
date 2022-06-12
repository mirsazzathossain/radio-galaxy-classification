from fileinput import filename
import os
import sys
import numpy as np
import pandas as pd
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from PIL import Image
import sys
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class MiraBestDataset(data.Dataset):
    """Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = 'batches'
    url = "http://www.jb.man.ac.uk/research/MiraBest/basic/MiraBest_basic_batches.tar.gz" 
    filename = "MiraBest_basic_batches.tar.gz"
    tgz_md5 = '6c9a3e6ca3c0f3d27f9f6dca1b9730e1'
    train_list = [
                  ['data_batch_1', '6c501a41da89217c7fda745b80c06e99'],
                  ['data_batch_2', 'e4a1e5d6f1a17c65a23b9a80969d70fb'],
                  ['data_batch_3', 'e326df6fe352b669da8bf394e8ac1644'],
                  ['data_batch_4', '7b9691912178497ad532c575e0132d1f'],
                  ['data_batch_5', 'de822b3c21f13c188d5fa0a08f9fcce2'],
                  ['data_batch_6', '39b38c3d63e595636509f5193a98d6eb'],
                  ['data_batch_7', 'f980bfd2b1b649f6142138f2ae76d087'],
                  ['data_batch_8', 'a5459294e551984ac26056ba9f69a3f8'],
                  ['data_batch_9', '34414bcae9a2431b42a7e1442cb5c73d'],
                  ]

    test_list = [
                 ['test_batch', 'd12d31f7e8d60a8d52419a57374d0095'],
                 ]
    meta = {
                'filename': 'batches.meta',
                'key': 'label_names',
                'md5': '97de0434158b529b5701bb3a1ed28ec6',
                }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.filename = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                self.filename.extend(entry['filenames'])

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img,(150,150))
        img = Image.fromarray(img,mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_filename(self):
        filename = [i.split('/')[-1] for i in self.filename]
        filename = [os.path.splitext(i)[0] for i in filename]
        return filename


def clean_data_table(data):
    '''
    Remove rows containing FR_class other than 100 and 200

    params:
        data: pandas dataframe
 
    Returns:
        pandas.DataFrame: Cleaned data table
    '''

    data = data[data['FR_class'].isin([100, 200])]
    return data


def create_catelog(mirabest_table, filenames):
    '''
    Combine the mirabest table and the filenames into a catelog

    Parameters
    ----------
    mirabest_table : pandas.DataFrame
        The mirabest table
    filenames : list
        The filenames of the mirabest table

    Returns
    -------
    catelog : pandas.DataFrame
        Full catelog of the mirabest table and the filenames
    '''

    catelog = mirabest_table.copy()

    filename = []
    index = []
    for i in range(len(mirabest_table)):
        search_text = f"{mirabest_table.iloc[i]['z']:06.4f}_{mirabest_table.iloc[i]['Size_rad']:07.2f}"

        parity = 0

        for j in range(len(filenames)):
            if search_text in filenames.iloc[j]['filename']:
                if parity == 0:
                    filename.append(filenames.iloc[j]['filename'])
                    index.append(j)
                    parity = 1
                else:
                    print(f"{search_text} is found in multiple files")

        if parity == 0:
            catelog.drop(i, axis=0, inplace=True)
            print(f"{search_text} is not found in any file")

    catelog['filename'] = filename
    catelog['index'] = index

    catelog.set_index('index', inplace=True)
    catelog.sort_index(inplace=True)
    catelog.reset_index(inplace=True)
    catelog.drop(columns=['index'], inplace=True)

    return catelog

def create_html(catelog, html_file):
    '''
    Create a pdf of the catelog

    Parameters
    ----------
    catelog : pandas.DataFrame
        The catelog
    html_file : str
        The path to the html file
    '''

    f = open(html_file, 'w')
    f.write(catelog.to_html())
    f.close()



if __name__ == "__main__":
    # download the dataset
    train_dataset = MiraBestDataset(root='./data', train=True, download=True)
    filename = train_dataset.get_filename()
    filenames = pd.DataFrame(filename, columns=['filename'])

    test_dataset = MiraBestDataset(root='./data', train=False, download=True)
    filename = test_dataset.get_filename()
    filenames = pd.concat([filenames, pd.DataFrame(filename, columns=['filename'])], ignore_index=True)

    filenames.to_csv('./data/filenames.csv', index=False)

    mirabest_table = pd.read_csv(r'data/table1.csv')
    mirabest_table = clean_data_table(mirabest_table)
    mirabest_table.to_csv('./data/table1_100_200.csv', index=False)
    
    mirabest_table = pd.read_csv (r'data/table1_100_200.csv')
    mirabest_filenames = pd.read_csv (r'data/filenames.csv')

    catelog = create_catelog(mirabest_table, mirabest_filenames)
    catelog.to_csv(r'data/catelog.csv')

    create_html(catelog, r'data/catelog.html')
