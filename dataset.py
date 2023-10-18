import base64
import os
import os.path as path
import random
import struct
import pickle
from collections import defaultdict
from glob import glob
from io import BytesIO
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from PIL import Image
from einops import unpack, rearrange, repeat, pack
from torch.utils.data import IterableDataset
from torchvision.datasets import CIFAR100, Omniglot
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm

from utils import Timer


class MetaCifar100(IterableDataset):
    meta_train_classes = None
    meta_test_classes = None
    data = None

    def __init__(self, config, root='./data', meta_split='train'):
        super().__init__()

        self.config = config
        self.root = root
        self.data_dir = path.join(root, 'cifar100')
        self.pickle_path = path.join(self.data_dir, 'cifar100.pickle')
        self.meta_split = meta_split

        if not path.exists(self.pickle_path):
            self.build_pickle()

        if self.data is None:
            with open(self.pickle_path, 'rb') as f, Timer('Pickle file loaded in {:.3f}s'):
                type(self).data = pickle.load(f)

        if self.meta_train_classes is None:
            classes = list(self.data.keys())
            random.seed(0)  # Make sure the same splits are used for all runs
            random.shuffle(classes)
            type(self).meta_train_classes = classes[config['meta_test_tasks']:]
            type(self).meta_test_classes = classes[:config['meta_test_tasks']]
            random.seed()  # Reset seed

        if self.meta_split == 'train':
            self.classes = self.meta_train_classes
        elif self.meta_split == 'test':
            self.classes = self.meta_test_classes
        else:
            raise ValueError('Unknown meta_split: {}'.format(self.meta_split))

    def __iter__(self):
        return self

    def __next__(self):
        classes = random.sample(self.classes, self.config['tasks'])

        # Sample examples for each class
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for cls_id, cls in enumerate(classes):
            imgs = random.sample(self.data[cls], self.config['train_shots'] + self.config['test_shots'])
            train_imgs = imgs[:self.config['train_shots']]
            test_imgs = imgs[self.config['train_shots']:]
            train_x.extend(train_imgs)
            train_y.extend([cls_id] * self.config['train_shots'])
            test_x.extend(test_imgs)
            test_y.extend([cls_id] * self.config['test_shots'])

        train_x = torch.tensor(np.stack(train_x))
        train_y = torch.tensor(train_y)
        test_x = torch.tensor(np.stack(test_x))
        test_y = torch.tensor(test_y)

        return train_x, train_y, test_x, test_y

    def build_pickle(self):
        splits = {
            'train': CIFAR100(path.join(self.data_dir), download=True, train=True),
            'test': CIFAR100(path.join(self.data_dir), download=True, train=False),
        }
        data = {}
        for split, cifar100 in splits.items():
            bchw = cifar100.data.transpose(0, 3, 1, 2)
            for x, y in tqdm(zip(bchw, cifar100.targets)):
                if y not in data:
                    data[y] = []
                data[y].append(x)
        with open(self.pickle_path + '.tmp', 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        os.rename(self.pickle_path + '.tmp', self.pickle_path)


class MetaOmniglot(IterableDataset):
    data = None

    def __init__(self, config, root='./data', meta_split='train'):
        super().__init__()

        self.config = config
        self.root = root
        self.data_dir = path.join(root, 'omniglot')
        self.pickle_path = path.join(self.data_dir, 'omniglot.pickle')
        self.meta_split = meta_split

        if not path.exists(self.pickle_path):
            print('Building pickle file...')
            self.build_pickle()

        if self.data is None:
            with open(self.pickle_path, 'rb') as f, Timer('Pickle file loaded in {:.3f}s'):
                type(self).data = pickle.load(f)

        self.classes = list(self.data[meta_split].keys())

        print('Decoding images...')
        self.split_data = {}
        for cls in tqdm(self.classes):
            cls_imgs = []
            for img_bytes in self.data[meta_split][cls]:
                img = Image.open(BytesIO(img_bytes)).convert('L')
                img = img.resize((self.config['x_h'], self.config['x_w']), resample=Image.BILINEAR)
                cls_imgs.append(pil_to_tensor(img))
            self.split_data[cls] = cls_imgs

    def __iter__(self):
        return self

    def __next__(self):
        # Sample a sequence of classes
        classes = random.sample(self.classes, self.config['tasks'])

        # Sample examples for each class
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for cls_id, cls in enumerate(classes):
            sampled_imgs = random.sample(
                self.split_data[cls], self.config['train_shots'] + self.config['test_shots'])
            train_imgs = sampled_imgs[:self.config['train_shots']]
            test_imgs = sampled_imgs[self.config['train_shots']:]
            train_x.extend(train_imgs)
            train_y.extend([cls_id] * self.config['train_shots'])
            test_x.extend(test_imgs)
            test_y.extend([cls_id] * self.config['test_shots'])

        train_x = torch.stack(train_x)
        train_y = torch.tensor(train_y)
        test_x = torch.stack(test_x)
        test_y = torch.tensor(test_y)

        return train_x, train_y, test_x, test_y

    def build_pickle(self):
        splits = {
            'train': Omniglot(self.data_dir, background=True, download=True),
            'test': Omniglot(self.data_dir, background=False, download=True)
        }
        data = {
            'train': {},
            'test': {}
        }
        for split, omniglot in splits.items():
            print(f'Building {split} split...')
            split_dict = data[split]
            for c, character in enumerate(tqdm(omniglot._characters)):
                split_dict[character] = []
                for i, (image_name, _) in enumerate(omniglot._character_images[c]):
                    with open(path.join(omniglot.target_folder, character, image_name), 'rb') as img_f:
                        img_bytes = img_f.read()
                    split_dict[character].append(img_bytes)
        with open(self.pickle_path + '.tmp', 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        os.rename(self.pickle_path + '.tmp', self.pickle_path)


class MetaCasia(IterableDataset):
    name = 'casia-hwdb'
    meta_train_classes = None
    meta_test_classes = None
    x_dict = None
    y_dict = None

    def __init__(self, config, root='./data', meta_split='train'):
        super().__init__()
        self.config = config
        self.root = root
        self.data_dir = path.join(root, self.name)
        self.meta_split = meta_split
        self.pickle_path = path.join(self.data_dir, f'{self.name}.pickle')

        if not path.exists(self.pickle_path):
            self.download()
            self.build_pickle()

        if self.x_dict is None:
            with open(self.pickle_path, 'rb') as pickle_file, Timer('Pickle file loaded in {:.3f}s'):
                type(self).x_dict, type(self).y_dict = pickle.load(pickle_file)

        if self.meta_train_classes is None:
            classes = list(self.x_dict.keys())
            random.seed(0)  # Make sure the same splits are used for all runs
            random.shuffle(classes)
            type(self).meta_train_classes = classes[config['meta_test_tasks']:]
            type(self).meta_test_classes = classes[:config['meta_test_tasks']]
            random.seed()  # Reset seed

        if self.meta_split == 'train':
            self.classes = self.meta_train_classes
        elif self.meta_split == 'test':
            self.classes = self.meta_test_classes
        else:
            raise ValueError('Unknown meta_split: {}'.format(self.meta_split))

        self.cache = {cls: {} for cls in self.classes}

    def __iter__(self):
        return self

    def __next__(self):
        # Sample a sequence of classes
        classes = random.sample(self.classes, self.config['tasks'])

        # Sample examples for each class
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for cls_id, cls in enumerate(classes):
            cls_imgs = self.x_dict[cls]
            cls_cache = self.cache[cls]
            sampled_indices = random.sample(
                range(len(cls_imgs)), self.config['train_shots'] + self.config['test_shots'])

            # Load sampled images
            imgs = []
            for idx in sampled_indices:
                if idx not in cls_cache:
                    img_bytes = cls_imgs[idx]
                    img = Image.open(BytesIO(img_bytes))
                    # img = img.resize((self.config['x_h'], self.config['x_w']), Image.BILINEAR)
                    img = pil_to_tensor(img)
                    cls_cache[idx] = img
                    cls_imgs[idx] = None
                imgs.append(cls_cache[idx])

            train_imgs = imgs[:self.config['train_shots']]
            test_imgs = imgs[self.config['train_shots']:]
            train_x.extend(train_imgs)
            train_y.extend([cls_id] * self.config['train_shots'])
            test_x.extend(test_imgs)
            test_y.extend([cls_id] * self.config['test_shots'])

        train_x = torch.stack(train_x)
        train_y = torch.tensor(train_y)
        test_x = torch.stack(test_x)
        test_y = torch.tensor(test_y)

        return train_x, train_y, test_x, test_y

    def download(self):
        download_links = [
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0TrainPart1.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0TrainPart2.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0TrainPart3.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0Test.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart1.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart2.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1Test.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.2TrainPart1.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.2TrainPart2.zip',
            'http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.2Test.zip'
        ]

        os.makedirs(self.data_dir, exist_ok=True)
        for link in download_links:
            file_name = link.split('/')[-1]
            download_path = path.join(self.data_dir, file_name)
            if not path.exists(download_path):
                os.system(f'wget -nc {link} -P {self.data_dir}')
            extract_path = download_path.replace('.zip', '')
            if not path.exists(extract_path):
                os.system(f'unzip {download_path} -d {extract_path + "_tmp"}')
                os.system(f'mv {extract_path + "_tmp"} {extract_path}')

    def build_pickle(self):
        gnt_files = sorted(glob(path.join(self.data_dir, 'Gnt*/*.gnt')))

        x_dict = {}
        y_dict = {}
        print(f'Converting {len(gnt_files)} *.gnt files to Python dictionary...')
        with Pool() as pool:
            for gnt_id, result in tqdm(pool.imap_unordered(process_gnt, gnt_files), total=len(gnt_files)):
                for i, (x, y) in enumerate(result):
                    if y in y_dict:
                        y_id = y_dict[y]
                    else:
                        y_id = len(y_dict)
                        y_dict[y] = y_id
                    if y_id not in x_dict:
                        x_dict[y_id] = []
                    x_dict[y_id].append(x)
        print(f'Saving Python dictionary to a pickle file...')
        with open(self.pickle_path + '.tmp', 'wb') as f:
            pickle.dump((x_dict, y_dict), f)
        os.rename(self.pickle_path + '.tmp', self.pickle_path)


def load_gnt_file(file_name):
    with open(file_name, 'rb') as f:
        while (packed_length := f.read(4)) != b'':
            # length = struct.unpack("<I", packed_length)[0]
            raw_label = struct.unpack("<cc", f.read(2))
            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]
            photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))

            label = str(raw_label[0] + raw_label[1], 'gbk')
            image = Image.fromarray(np.array(photo_bytes, dtype=np.uint8).reshape(height, width))

            yield image, label


def resize_image(image, size):
    width, height = image.size
    if width > height:
        new_width = size
        new_height = round((size * height) / width)
    else:
        new_height = size
        new_width = round((size * width) / height)
    resized_image = image.resize((new_width, new_height))
    background = Image.new('L', (size, size), (255,))
    offset = ((size - new_width) // 2, (size - new_height) // 2)
    background.paste(resized_image, offset)
    return background


def process_gnt(gnt_file):
    gnt_id, ext = path.splitext(path.basename(gnt_file))
    result = []
    for i, (x, y) in enumerate(load_gnt_file(gnt_file)):
        if 0 in x.size:
            print(f'Skipping image {i} in {gnt_file} size: {x.size})')
            continue
        img = resize_image(x, 32)
        bio = BytesIO()
        img.save(bio, format='png')
        bio.seek(0)
        img_bytes = bio.read()
        result.append((img_bytes, y))
    return gnt_id, result


class MetaCasiaCompletion(MetaCasia):
    def __next__(self):
        # Get 32x32 images
        train_x, train_y, test_x, test_y = super().__next__()

        # Split x into two 16x32 images
        train_x, train_y = unpack(train_x, [[self.config['x_h']], [self.config['x_h']]], 'n c * w')
        test_x, test_y = unpack(test_x, [[self.config['x_h']], [self.config['x_h']]], 'n c * w')
        train_y = rearrange(train_y, 'n c h w -> n (c h w)')
        test_y = rearrange(test_y, 'n c h w -> n (c h w)')
        return train_x, train_y, test_x, test_y


class MetaCasiaRotation(MetaCasia):
    def __init__(self, config, root='./data', meta_split='train'):
        super().__init__(config, root=root, meta_split=meta_split)

    def __next__(self):
        # Sample a sequence of classes
        classes = random.sample(self.classes, self.config['tasks'])

        # Sample examples for each class
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for cls_id, cls in enumerate(classes):
            cls_imgs = self.x_dict[cls]
            cls_cache = self.cache[cls]

            # Sample rotation angles
            offset = random.random()  # prevent meta-learning a general rotational pattern
            angles = 360 * (np.random.rand(self.config['train_shots'] + self.config['test_shots']) + offset)
            rads = angles * np.pi / 180
            cos_sin = np.stack([np.cos(rads), np.sin(rads)], axis=1)
            train_y.append(cos_sin[:self.config['train_shots']])
            test_y.append(cos_sin[self.config['train_shots']:])

            sampled_indices = random.sample(
                range(len(cls_imgs)), self.config['train_shots'] + self.config['test_shots'])
            # Load sampled images
            imgs = []
            for idx, angle in zip(sampled_indices, angles):
                if idx in cls_cache:
                    img = cls_cache[idx]
                else:
                    img_bytes = cls_imgs[idx]
                    img = Image.open(BytesIO(img_bytes))
                    # img = img.resize((self.config['x_h'], self.config['x_w']), Image.BILINEAR)
                    cls_cache[idx] = img

                img = img.rotate(angle, fillcolor=255)
                img = pil_to_tensor(img)
                imgs.append(img)

            train_imgs = imgs[:self.config['train_shots']]
            test_imgs = imgs[self.config['train_shots']:]
            train_x.extend(train_imgs)
            test_x.extend(test_imgs)

        train_x = torch.stack(train_x)
        test_x = torch.stack(test_x)
        train_y = torch.tensor(pack(train_y, '* d')[0], dtype=torch.float)
        test_y = torch.tensor(pack(test_y, '* d')[0], dtype=torch.float)

        return train_x, train_y, test_x, test_y


class MetaMsCeleb1M(MetaCasia):
    name = 'MS-Celeb-1M'

    def __init__(self, config, root='./data', meta_split='train'):
        self.tsv_path = path.join(root, self.name, 'data/aligned_face_images/FaceImageCroppedWithAlignment.tsv')
        super().__init__(config, root, meta_split)

    def download(self):
        if not path.exists(self.tsv_path):
            raise RuntimeError(f'Please download {self.name} dataset manually, following the instructions in README.md')

    def build_pickle(self):
        print(f'Counting number of images per class...')
        img_counts = defaultdict(int)
        with open(self.tsv_path, 'r') as f:
            for line in tqdm(f):
                fields = line.strip().split('\t')
                img_counts[fields[0]] += 1
        total_samples = sum(img_counts.values())
        # Skip classes with less than 20 images
        too_few = set(key for key, count in img_counts.items() if count < 20)

        print(f'Converting to Python dictionary...')
        x_dict = {}
        y_dict = {}
        with open(self.tsv_path, 'r') as f:
            for line in tqdm(f, total=total_samples):
                fields = line.strip().split('\t')
                y = fields[0]
                if y in too_few:
                    continue

                if y in y_dict:
                    y_id = y_dict[y]
                else:
                    # New class
                    y_id = len(y_dict)
                    y_dict[y] = y_id
                    x_dict[y_id] = []

                # Parse image and save as PNG binary
                imgbase64 = fields[-1]
                imgdata = base64.b64decode(imgbase64)
                img_array = np.frombuffer(imgdata, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                resized = cv2.resize(img, (32, 32))
                pil_img = Image.fromarray(resized[:, :, ::-1])
                bio = BytesIO()
                pil_img.save(bio, format='PNG')
                bio.seek(0)
                img_bytes = bio.read()
                x_dict[y_id].append(img_bytes)

        with open(self.pickle_path + '.tmp', 'wb') as f:
            pickle.dump((x_dict, y_dict), f)
        os.rename(self.pickle_path + '.tmp', self.pickle_path)


class Sine(IterableDataset):
    def __init__(self, config, root=None, meta_split=None):
        super().__init__()
        self.config = config
        self.x_t = np.linspace(0, 10, config['x_dim']).reshape(1, 1, -1)
        self.y_t = np.linspace(0, 10, config['y_dim']).reshape(1, 1, -1)

    def __iter__(self):
        return self

    def __next__(self):
        tasks = self.config['tasks']
        shots = self.config['train_shots'] + self.config['test_shots']
        freq = np.random.rand(tasks, 1, 1) + 0.1
        pi2 = 2 * np.pi
        x_phase = np.random.rand(tasks, 1, 1) * pi2
        y_phase = np.random.rand(tasks, 1, 1) * pi2
        train_amp = np.random.rand(tasks, self.config['train_shots'], 1) + 0.5
        test_amp = np.random.rand(tasks, self.config['test_shots'], 1) + 0.5

        train_x = train_amp * np.sin(pi2 * freq * self.x_t + x_phase)
        train_y = train_amp * np.sin(pi2 * freq * self.y_t + y_phase)
        test_x = test_amp * np.sin(pi2 * freq * self.x_t + x_phase)
        test_y = test_amp * np.sin(pi2 * freq * self.y_t + y_phase)

        # Add noise to x
        train_x_noise = np.random.normal(0, 0.1, train_x.shape)
        test_x_noise = np.random.normal(0, 0.1, test_x.shape)
        train_x += train_x_noise
        test_x += test_x_noise

        train_x = rearrange(train_x, 't s d -> (t s) d')
        train_y = rearrange(train_y, 't s d -> (t s) d')
        test_x = rearrange(test_x, 't s d -> (t s) d')
        test_y = rearrange(test_y, 't s d -> (t s) d')

        return torch.tensor(train_x, dtype=torch.float), \
            torch.tensor(train_y, dtype=torch.float), \
            torch.tensor(test_x, dtype=torch.float), \
            torch.tensor(test_y, dtype=torch.float)


DATASET = {
    'cifar100': MetaCifar100,
    'omniglot': MetaOmniglot,
    'casia': MetaCasia,
    'casia_comp': MetaCasiaCompletion,
    'casia_rot': MetaCasiaRotation,
    'celeb': MetaMsCeleb1M,
    'sine': Sine,
}
