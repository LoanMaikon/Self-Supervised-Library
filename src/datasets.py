from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from glob import glob
from math import ceil
import scipy.io
import os

'''
transforms: list of transforms to apply to the same image.
'''
class datasets(Dataset):
    def __init__(self, operation, datasets_folder_path, dataset_name, separate_val_subset, val_size, transforms, times):
        super(datasets, self).__init__()

        self.operation = operation
        self.datasets_folder_path = datasets_folder_path
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.separate_val_subset = separate_val_subset
        self.val_size = val_size
        self.times = times

        self.images = []
        self.labels = []

        match dataset_name:
            case "imagenet":
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

        returns = []
        for it, transform in enumerate(self.transforms):
            for _ in range(self.times[it]):
                returns.append(transform(image))

        return returns, self.labels[idx]
