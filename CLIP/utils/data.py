import os
import os.path as osp
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from utils.toolkit import split_images_labels, get_dataset_class_names
from clip_backbones.clip import clip
import json
import yaml
import warnings


base_dir = '/data/Datasets/'


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        # train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        # test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        train_dataset = datasets.cifar.CIFAR10(os.path.join(base_dir, 'cifarpy'), train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(os.path.join(base_dir, 'cifarpy'), train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        try:
            train_dataset = datasets.cifar.CIFAR100(os.path.join(base_dir, 'cifarpy'), train=True, download=True)
            test_dataset = datasets.cifar.CIFAR100(os.path.join(base_dir, 'cifarpy'), train=False, download=True)
        except:
            data_path = os.path.join(os.path.dirname(os.getcwd()), 'train_data')
            train_dataset = datasets.cifar.CIFAR100(data_path, train=True, download=True)  # for gadi
            test_dataset = datasets.cifar.CIFAR100(data_path, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t


def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t


class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if 'vl' in args['model_name']:
            self.train_trsf = clip._transform(224, is_train=True)
            self.test_trsf = clip._transform(224, is_train=False)
        elif args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()
        self.class_names = None

    def download_data(self):
        try:
            base_dir = '/data/Datasets/'
            train_dataset = datasets.cifar.CIFAR100(os.path.join(base_dir, 'cifarpy'), train=True, download=True)
            test_dataset = datasets.cifar.CIFAR100(os.path.join(base_dir, 'cifarpy'), train=False, download=True)
        except:
            data_path = os.path.join(os.path.dirname(os.getcwd()), 'train_data')
            train_dataset = datasets.cifar.CIFAR100(data_path, train=True, download=True)
            test_dataset = datasets.cifar.CIFAR100(data_path, train=False, download=True)

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        self.class_names = train_dataset.classes


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()
        
        self.class_names = None

    def download_data(self):
        if self.args['gadi']:
            # base_dir = './data/'
            base_dir = os.path.dirname(os.getcwd())
        else:
            base_dir = '/data/Datasets/'

        # assert 0, "You should specify the folder of your dataset"
        # train_dir = "./data/imagenet-r/train/"
        # test_dir = "./data/imagenet-r/test/"
        train_dir = os.path.join(base_dir, 'imagenet-r/train')
        test_dir = os.path.join(base_dir, 'imagenet-r/test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        self.class_names = get_dataset_class_names(base_dir, 'imagenet-r')
        print(self.class_names)


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class CUB(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()
        
        self.class_names = None

    def download_data(self):
        if self.args['gadi']:
            # base_dir = './data/'
            base_dir = os.path.dirname(os.getcwd())
        else:
            base_dir = '/data/Datasets/'
        # train_dir = "./data/cub/train/"
        # test_dir = "./data/cub/test/"
        train_dir = os.path.join(base_dir, 'cub/train')
        test_dir = os.path.join(base_dir, 'cub/test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        self.class_names = load_json('utils/labels.json')['cub']
        print(self.class_names)
        

class TinyIMN(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()
        
        self.class_names = None

    def download_data(self):
        if self.args['gadi']:
            base_dir = './data/'
        else:
            base_dir = '/data/Datasets/'
        # train_dir = "./data/cub/train/"
        # test_dir = "./data/cub/test/"
        train_dir = os.path.join(base_dir, 'tiny-imagenet-200')
        test_dir = os.path.join(base_dir, 'tiny-imagenet-200')
        self.class_names = load_json('utils/labels.json')['tinyimagenet']
        print(self.class_names)
        
        wnids_file = "/data/Datasets/tiny-imagenet-200/wnids.txt"
        with open(wnids_file, "r") as f:
            class_ids = sorted([line.strip() for line in f.readlines()])  # Sort WNIDs
        # print(f"Total classes: {len(class_ids)}")
        # print(f"Example sorted class IDs: {class_ids[:5]}")
        # Load WordNet ID to class name mapping
        words_file = "/data/Datasets/tiny-imagenet-200/words.txt"
        wnid_to_classname = {}
        with open(words_file, "r") as f:
            for line in f.readlines():
                wnid, classname = line.strip().split("\t")
                wnid_to_classname[wnid] = classname
        # Get class names for Tiny ImageNet
        tiny_imagenet_classes = {wnid: wnid_to_classname[wnid] for wnid in class_ids}
        class_to_idx = {name: i for i, name in enumerate(tiny_imagenet_classes)}

        # train_dset = datasets.ImageFolder(train_dir)
        # test_dset = datasets.ImageFolder(test_dir)

        # self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        # self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.train_data = []
        self.train_targets = []
        for class_name in os.listdir(os.path.join(train_dir, "train")):
            class_path = os.path.join(train_dir, "train", class_name, "images")
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.train_data.append(img_path)
                self.train_targets.append(class_to_idx[class_name])
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        
        self.test_data = []
        self.test_targets = []
        val_labels = {}
        with open(os.path.join(test_dir, "val", "val_annotations.txt"), "r") as f:
            for line in f.readlines():
                parts = line.split("\t")
                val_labels[parts[0]] = class_to_idx[parts[1]]

        val_images_path = os.path.join(test_dir, "val", "images")
        for img_name in os.listdir(val_images_path):
            img_path = os.path.join(val_images_path, img_name)
            if img_name in val_labels:
                self.test_data.append(img_path)
                self.test_targets.append(val_labels[img_name])
                
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)


class DomainNet(iData):
    def __init__(self, args):
        self.args = args
        self.use_path = True
        
        class_order = np.arange(345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]
        self.class_names = None
        from utils.class_names import domainnet_classnames
        self.class_names = list(domainnet_classnames.values())
        for i in range(len(self.class_names)):
            self.class_names[i] = self.class_names[i].replace('_', ' ')
        print(self.class_names)

    def download_data(self):      
        # train_data_config = yaml.load(open('utils/domainnet/domainnet_train.yaml', 'r'), Loader=yaml.Loader)
        # test_data_config = yaml.load(open('utils/domainnet/domainnet_test.yaml', 'r'), Loader=yaml.Loader)
        # self.train_data = np.array(train_data_config['data'])
        # self.train_targets = np.array(train_data_config['targets'])
        # self.test_data = np.array(test_data_config['data'])
        # self.test_targets = np.array(test_data_config['targets'])
        if self.args['gadi']:
            base_dir = './data/'
        else:
            base_dir = '/data/Datasets/'

        
        self.image_list_root = os.path.join(base_dir, 'domainnet/')

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            # imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
            imgs += [(val.split()[0], int(val.split()[1])) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            # imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
            imgs += [(val.split()[0], int(val.split()[1])) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.test_data = np.array(train_x)
        self.test_targets = np.array(train_y)

        
class UCF(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()
        
        self.class_names = None

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        # train_dir = "./data/cub/train/"
        # test_dir = "./data/cub/test/"
        train_dir = os.path.join(base_dir, 'ucf101/train')
        test_dir = os.path.join(base_dir, 'ucf101/test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        # self.class_names = load_json('utils/labels.json')['ucf101']
        self.class_names = list(train_dset.class_to_idx.keys())
        self.class_names = [self.class_names[i].replace('_', ' ') for i in range(len(self.class_names))]
        # print(self.class_names)
        # print(list(train_dset.class_to_idx.keys()))
        # assert self.class_names == list(train_dset.class_to_idx.keys())
        


class AirCraft(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()
        
        self.class_names = None

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        if self.args['gadi']:
            base_dir = './data/'
        else:
            base_dir = '/data/Datasets/'
        # train_dir = "./data/cub/train/"
        # test_dir = "./data/cub/test/"
        train_dir = os.path.join(base_dir, 'aircraft/train')
        test_dir = os.path.join(base_dir, 'aircraft/test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        
        # print(train_dset.class_to_idx)
        # print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        self.class_names = load_json('utils/labels.json')['aircraft']
        # self.class_names = list(train_dset.class_to_idx.keys())
        print(self.class_names)
        # assert train_dset.class_to_idx.keys() == self.class_names


class Cars(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()
        
        self.class_names = None

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        # train_dir = "./data/cub/train/"
        # test_dir = "./data/cub/test/"
        if self.args['gadi']:
            base_dir = './data/'
        else:
            base_dir = '/data/Datasets/'
        train_dir = os.path.join(base_dir, 'cars/train')
        test_dir = os.path.join(base_dir, 'cars/test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        
        # print(train_dset.class_to_idx)
        # print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        # self.class_names = load_json('utils/labels.json')['cars']
        print(list(train_dset.class_to_idx.keys()))
        self.class_names = list(train_dset.class_to_idx.keys())
        print(self.class_names)
        # assert train_dset.class_to_idx.keys() == self.class_names
        

class ISIC(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]
        
        # SIZE = (224, 224)
        # MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # self.train_trsf = transforms.Compose([
        #     transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        #     transforms.RandomCrop(SIZE[0]),
        #     transforms.RandomHorizontalFlip(0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=MEAN, std=STD),
        # ])

        # self.test_trsf = transforms.Compose([
        #     transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(SIZE[0]),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=MEAN, std=STD),
        # ])


        # self.class_order = np.arange(7).tolist()
        self.class_order = np.arange(6).tolist()
        
        # self.class_names = ['melanoma',
        #                     'melanocytic nevus',
        #                     'basal cell carcinoma',
        #                     'actinic keratosis or intraepithelial carcinoma',
        #                     'benign keratosis',
        #                     'dermatofibroma',
        #                     'vascular skin lesion']
        self.class_names = ['melanoma',
                            'basal cell carcinoma',
                            'actinic keratosis or intraepithelial carcinoma',
                            'benign keratosis',
                            'dermatofibroma',
                            'vascular skin lesion']
        
    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        # train_dir = "./data/cub/train/"
        # test_dir = "./data/cub/test/"
        from utils.isic_loader import Isic
        
        if self.args['gadi']:
            base_dir = './data'
        else:
            base_dir = '/data/Datasets'
            
        base_dir = os.path.join(base_dir, 'isic')
            
        train_data = Isic(
            root=base_dir,
            train=True,
            # download=True,
            download=False,
            # transform=self.train_trsf
        )
        test_data = Isic(
            root=base_dir,
            train=False,
            # download=True,
            download=False,
            # transform=self.test_trsf
        )
        
        self.train_data, self.train_targets = train_data.data, np.array(
            train_data.targets
        )
        self.test_data, self.test_targets = test_data.data, np.array(
            test_data.targets
        )
        
        # self.train_data = train_data.data
        # self.train_targets = train_data.targets
        # self.test_data = test_data.data
        # self.test_targets = test_data.targets
        print(self.class_names)
        

class Caltech101(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()
        
        self.class_names = None

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        # train_dir = "./data/cub/train/"
        # test_dir = "./data/cub/test/"
        train_dir = os.path.join(base_dir, 'caltech-101/train')
        test_dir = os.path.join(base_dir, 'caltech-101/test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        
        # print(train_dset.class_to_idx)
        # print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        self.class_names = load_json('utils/labels.json')['caltech101']
        # self.class_names = list(train_dset.class_to_idx.keys())
        print(train_dset.class_to_idx.keys())
        print(self.class_names)
        # assert train_dset.class_to_idx.keys() == self.class_names


class Caltech101(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()
        
        self.class_names = None

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        # train_dir = "./data/cub/train/"
        # test_dir = "./data/cub/test/"
        train_dir = os.path.join(base_dir, 'caltech-101/train')
        test_dir = os.path.join(base_dir, 'caltech-101/test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        
        # print(train_dset.class_to_idx)
        # print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        self.class_names = load_json('utils/labels.json')['caltech101']
        # self.class_names = list(train_dset.class_to_idx.keys())
        print(train_dset.class_to_idx.keys())
        print(self.class_names)
        # assert train_dset.class_to_idx.keys() == self.class_names


class ObjectNet(iData):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()
        
        self.class_names = None

    def download_data(self):
        train_dir = os.path.join(base_dir, 'objectnet/train')
        test_dir = os.path.join(base_dir, 'objectnet/test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        self.class_names = list(train_dset.class_to_idx.keys())
        self.class_names = [self.class_names[i].replace('_', ' ') for i in range(len(self.class_names))]
        # self.class_names = list(train_dset.class_to_idx.keys())
        # print(train_dset.class_to_idx.keys())
        print(self.class_names)
        # assert train_dset.class_to_idx.keys() == self.class_names


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/vtab-cil/vtab/train/"
        test_dir = "./data/vtab-cil/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
