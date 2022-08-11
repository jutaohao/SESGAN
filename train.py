"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES
import sys

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model

##
def train_cifar10():
    """ Training
    """
    opt = Options().parse()
    opt.dataset = "cifar10"
    t = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,'truck': 9}
    # t = {'bird': 2}
    opt.nc = 3
    for key in t:
        opt.normal_class=key
        data = load_data(opt)
        model = load_model(opt, data)
        model.train()

def train_MNIST():
    """ Training
    """
    opt = Options().parse()
    opt.dataset = "mnist"
    opt.nc = 1
    t = {'0': 0,'1':1,'2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9}
    for key in t:
        opt.normal_class=key
        data = load_data(opt)
        model = load_model(opt, data)
        model.train()
def train_GTSRB():
    """ Training
    """
    opt = Options().parse()
    opt.dataset = "GTSRB"
    opt.nc = 3

    opt.normal_class="stop_sign"
    data = load_data(opt)
    model = load_model(opt, data)
    model.train()

if __name__ == '__main__':
    opt = Options().parse()
    dataset = opt.dataset
    if dataset.lower() == 'cifar10':
        train_cifar10()
    elif dataset.lower() == 'minist':
        train_MNIST()
    else:
        train_GTSRB()

    # train_MNIST()
    # train_GTSRB()