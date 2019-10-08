from imports import *

class DirectionsDataset(Dataset):
    """Directions dataset."""

    def __init__(self, csv_file, root_dir, transform_img=None, transform_lab=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform_img = transform_img
        self.transform_lab = transform_lab

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.label.iloc[idx, 0])
        image = io.imread(img_name+'.jpg')
        sample = image
        label = self.label.iloc[idx, 1]

        if self.transform_img:
            sample = self.transform_img(sample)

        return sample, label


class BboxDataset(Dataset):
    """Bbox dataset."""

    def __init__(self, csv_file, root_dir, transform=None, transform_lab=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sz = 224

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.label.iloc[idx, 0])
        image = io.imread(img_name)
        sample = image
        
        h, w = sample.shape[:2]; new_h, new_w = (224,224)
        bb = np.array([float(x) for x in self.label.iloc[idx, 1].split(' ')], dtype=np.float32)
        bb = np.reshape(bb, (int(bb.shape[0]/2),2))
        bb = bb * [new_h / h, new_w / w]
        bb = bb.flatten()
        bb = T(np.concatenate((np.zeros((189*4) - len(bb)), bb), axis=None))

        if self.transform:
            sample = self.transform(sample)

        return sample, bb


class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2):
        self.ds,self.y2 = ds,y2
        self.sz = ds.sz
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        #set_trace()
        self.y2[i] = np.concatenate((np.zeros(189 - len(self.y2[i])), self.y2[i]), axis=None) # 15 max num of classes
        x,y = self.ds[i]
        return (x, (y,self.y2[i]))
