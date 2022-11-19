import torch
import os
import cv2
import numpy as np

class WSI(torch.utils.data.Dataset):
    def __init__(self, label_file_path=None, dataset_path=None):
        super(WSI, self).__init__()
        self.dataset_path = dataset_path
        self.dataset_name, self.class_label_list = [], []
        with open(label_file_path, 'r') as listFile:
            for line in listFile:
                name, class_label = line[:-1].split(":")
                self.dataset_name.append(name)
                self.class_label_list.append(class_label)
    
    def img_init_process(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.array(x).astype('float32') / 255
        x = np.transpose(x, (2, 0, 1))
        return x
    
    def __len__(self):
        return len(self.dataset_name)

    def __getitem__(self, idx):
        image_name = self.dataset_name[idx]
        image = cv2.imread(os.path.join(self.dataset_path, image_name), cv2.IMREAD_COLOR)
        image = self.img_init_process(image)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        class_label = self.class_label_list[idx]
        class_label = torch.from_numpy(class_label).type(torch.FloatTensor)
        return image, class_label
