import torch
import os
import cv2
import numpy as np


class WSIData(torch.utils.data.Dataset):
    def __init__(self, dataset_path=None):
        super(WSIData, self).__init__()
        self.dataset_path = dataset_path
        self.dataset_name, self.class_label_list = [], []

        filenames = os.listdir(dataset_path)
        for file in filenames:
            self.dataset_name.append(file)
            self.class_label_list.append(float(file[3]))
        
        self.class_label_list = np.array(self.class_label_list)
        self.class_label_list = self.class_label_list.astype('float').reshape(-1, 1)
    
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
        class_label = torch.from_numpy(class_label).type(torch.LongTensor)
        return image, class_label
