import os
import math
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image


def make_dataset(att_list_file, mode, selected_attrs):
    assert mode in ['train', 'val', 'test']
    lines = [line.rstrip() for line in open(att_list_file, 'r')]
    all_attr_names = lines[1].split()
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[2:]
    if mode == 'train':
        lines = lines[:182000]
    if mode == 'val':
        lines = lines[182000:182637]
    if mode == 'test':
        lines = lines[:]  # test all

    items = []
    for i, line in enumerate(lines):
        split = line.split()
        # filename = split[0].replace('jpg', 'png')
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')  # change label '-1'/'1' to 0/1
        items.append([filename, label])
    return items


class CelebADataset(data.Dataset):
    def __init__(self, datadir, att_list_file, mode, selected_attrs,
                 transform=None):
        self.items = make_dataset(att_list_file, mode, selected_attrs)
        self.root = datadir
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        filename, label = self.items[index]
        image = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), filename

    def __len__(self):
        return len(self.items)


class CelebADataLoader(object):
    def __init__(self, datadir, att_list_file, mode, selected_attrs,
                 crop_size=None, image_size=128, batch_size=16,
                 num_workers=8):
        if mode not in ['train', 'test', ]:
            return

        transform = []
        if crop_size is not None:
            transform.append(transforms.CenterCrop(crop_size))
        transform.append(transforms.Resize(image_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))  # value range -1~1

        if mode == 'train':
            # make val loader before transform is inserted
            val_transform = transforms.Compose(transform)
            val_set = CelebADataset(
                datadir, att_list_file, 'val', selected_attrs,
                transform=val_transform)
            self.val_loader = data.DataLoader(
                val_set, batch_size=batch_size, shuffle=False,
                num_workers=num_workers)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            transform.insert(0, transforms.RandomHorizontalFlip())
            train_transform = transforms.Compose(transform)
            train_set = CelebADataset(
                datadir, att_list_file, 'train', selected_attrs,
                transform=train_transform)
            self.train_loader = data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=num_workers)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            test_transform = transforms.Compose(transform)
            test_set = CelebADataset(
                datadir, att_list_file, 'test', selected_attrs,
                transform=test_transform)
            self.test_loader = data.DataLoader(
                test_set, batch_size=1, shuffle=False,
                num_workers=num_workers)
            self.test_iterations = int(math.ceil(len(test_set)))
