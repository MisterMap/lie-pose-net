"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
pytorch data loader for the 7-scenes dataset
"""

import os
import os.path as osp

import numpy as np
from torch.utils import data

from .utils import load_image


class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7, skip_images=False,):
        """
        :param scene: scene name ['chess', 'pumpkin', ...]
        :param data_path: root 7scenes data directory.
        Usually '../data/7scenes'
        :param train: if True, return the training images. If False, returns the
        testing images
        :param transform: transform to apply to the images
        :param target_transform: transform to apply to the positions
        :param mode: 0: just color image, 1: just depth image, 2: [c_img, d_img]
        :param skip_images: If True, skip loading images and return None instead
        """
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)

        # directories
        base_directory = osp.join(osp.expanduser(data_path), scene)

        # decide which sequences to use
        if train:
            split_file = osp.join(base_directory, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_directory, 'TestSplit.txt')
        with open(split_file, 'r') as fd:
            sequences = [int(x.split('sequence')[-1]) for x in fd if not x.startswith('#')]

        # read positions and collect image names
        self.color_images = []
        self.depth_images = []
        self.positions = []
        for sequence in sequences:
            sequence_directory = osp.join(base_directory, 'seq-{:02d}'.format(sequence))
            if not osp.isdir(sequence_directory):
                self.unzip_sequence(base_directory, sequence)
            pose_filenames = [x for x in os.listdir(osp.join(sequence_directory, '.')) if x.find('pose') >= 0]

            frame_indexes = np.arange(len(pose_filenames), dtype=np.int)
            positions = [np.loadtxt(osp.join(sequence_directory, 'frame-{:06d}.pose.txt'.
                                             format(i))) for i in frame_indexes]
            color_images = [osp.join(sequence_directory, 'frame-{:06d}.color.png'.format(i))
                            for i in frame_indexes]
            depth_images = [osp.join(sequence_directory, 'frame-{:06d}.depth.png'.format(i))
                            for i in frame_indexes]
            self.color_images.extend(color_images)
            self.depth_images.extend(depth_images)
            self.positions.extend(positions)
        self.positions = np.array(self.positions)

    @staticmethod
    def unzip_sequence(base_directory, sequence):
        import zipfile
        sequence_directory = osp.join(base_directory, 'seq-{:02d}'.format(sequence))
        zip_file = osp.join(base_directory, 'seq-{:02d}.zip'.format(sequence))
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(sequence_directory)

    def __getitem__(self, index):
        pose = None
        if self.skip_images:
            image = None
            pose = self.positions[index]
        elif self.mode == 0:
            image = None
            while image is None:
                image = load_image(self.color_images[index])
                pose = self.positions[index]
                index += 1
            index -= 1
        elif self.mode == 1:
            image = None
            while image is None:
                image = load_image(self.depth_images[index])
                pose = self.positions[index]
                index += 1
            index -= 1
        elif self.mode == 2:
            c_img = None
            d_img = None
            while (c_img is None) or (d_img is None):
                c_img = load_image(self.color_images[index])
                d_img = load_image(self.depth_images[index])
                pose = self.positions[index]
                index += 1
            image = [c_img, d_img]
            index -= 1
        else:
            raise Exception('Wrong mode {:d}'.format(self.mode))

        pose = pose.astype(np.float32)
        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return image, pose

        if self.transform is not None:
            if self.mode == 2:
                image = [self.transform(i) for i in image]
            else:
                image = self.transform(image)

        return {"image": image,
                "position": pose}

    def __len__(self):
        return self.positions.shape[0]
