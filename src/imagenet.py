from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from glob import glob
from math import ceil
import os
import torchvision.transforms as v2
import scipy.io

class imagenet(Dataset):
    def __init__(self, operation, datasets_folder_path, dataset_name, transform, separate_val_subset, val_size, apply_data_augmentation):
        super(imagenet, self).__init__()

        self.operation = operation
        self.datasets_folder_path = datasets_folder_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.separate_val_subset = separate_val_subset
        self.val_size = val_size
        self.apply_data_augmentation = apply_data_augmentation

        self.images = []
        self.labels = []

        validation_gd_path = f"{self.datasets_folder_path}{self.dataset_name}/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        meta_path = f"{self.datasets_folder_path}{self.dataset_name}/ILSVRC2012_devkit_t12/data/meta.mat"

        data = scipy.io.loadmat(open(meta_path, "rb"))
        synsets = data.get('synsets')

        id_to_wnid = {}
        wnid_to_class = {}
        for synset in synsets:
            if int(synset['num_children'][0][0][0]) == 0:
                wnid = synset['WNID'][0][0]
                words = synset['words'][0][0]

                wnid_to_class[wnid] = words
                id_to_wnid[int(synset['ILSVRC2012_ID'][0][0][0])] = wnid

        classes = sorted(wnid_to_class.values())
        class_to_id = {cls: idx for idx, cls in enumerate(classes)}

        train_images_per_class = {}
        train_wnid = os.listdir(self.datasets_folder_path + self.dataset_name + "/train/")
        for wnid in train_wnid:
            _images = glob(self.datasets_folder_path + self.dataset_name + "/train/" + wnid + "/*.JPEG")
            train_images_per_class[wnid] = _images

        if self.separate_val_subset:
            val_images_per_class = {}

            for wnid, _images in train_images_per_class.items():
                n_val_images = ceil(len(_images) * self.val_size)

                val_images_per_class[wnid] = _images[-n_val_images:]
                train_images_per_class[wnid] = _images[:-n_val_images]

        match self.operation:
            case "train":
                for wnid, _images in train_images_per_class.items():
                    self.images.extend(_images)
                    class_name = wnid_to_class[wnid]
                    class_id = class_to_id[class_name]
                    self.labels.extend([class_id] * len(_images))

            case "val":
                if self.separate_val_subset:
                    for wnid, _images in val_images_per_class.items():
                        self.images.extend(_images)
                        class_name = wnid_to_class[wnid]
                        class_id = class_to_id[class_name]
                        self.labels.extend([class_id] * len(_images))
                else:
                    raise ValueError("separate_val_subset is false in config file")

            case "test":
                val_images = sorted(glob(self.datasets_folder_path + self.dataset_name + "/val/*.JPEG"))

                idx_to_wnid = {}
                with open(validation_gd_path, "r") as file:
                    for idx, line in enumerate(file):
                        line = line.strip()
                        if not line:
                            continue
                        wnid = id_to_wnid[int(line)]
                        idx_to_wnid[idx] = wnid
                
                for idx, image in enumerate(val_images):
                    wnid = idx_to_wnid[idx]

                    self.images.append(image)
                    class_name = wnid_to_class[wnid]
                    class_id = class_to_id[class_name]
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        image = read_image(image_path, ImageReadMode.RGB)

        if self.apply_data_augmentation:
            x1 = self.transform(image)
            x2 = self.transform(image)

            return x1, x2
        
        image = self.transform(image)

        return image, self.labels[idx]
