from huva.th_util import Flatten, MonitoredAdam, MonitoredSGD, MonitoredRMSprop, \
     get_model_param_norm, get_num_correct, set_learning_rate, init_weights, \
     get_layer_utilization, get_all_utilization
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision
from collections import OrderedDict
from pprint import pprint
import gc
from pympler.tracker import SummaryTracker
from multi_gpu import wrap_model

"""
TODOs:
1. Figure out ordinary dropout
2. figure out (increase eps, separate weight decay) actually improve things
3. (remove 0-unit, remove low-quantile unit)
"""

model_conf_11 = [
    ('conv1_1', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, None)),
    ('conv3_2', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (256, None)),
    ('conv4_2', (256, None)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (160, None)),
    ('conv5_2', (160, None)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (160, None)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]

model_conf_16 = [
    ('conv1_1', (64,  0.3)),
    ('conv1_2', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, 0.4)),
    ('conv2_2', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, 0.4)),
    ('conv3_2', (256, 0.4)),
    ('conv3_3', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (512, 0.4)),
    ('conv4_2', (512, 0.4)),
    ('conv4_3', (512, None)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (512, 0.4)),
    ('conv5_2', (512, 0.4)),
    ('conv5_3', (512, None)),
    ('pool5'  , (2,   2)),
    ('drop5'  , (None,0.5)),
    ('fc6'    , (512, 0.5)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]

model_conf = model_conf_16

def make_model(num_class=10):
    in_channel = 3
    """ early layers """
    layers = OrderedDict()
    dropout = nn.Dropout # nn.Dropout2d
    for name, info in model_conf:
        if name.startswith('conv') or name.startswith('fc'):
            num_chan, drop_p = info
            k,pad = (3,1) if name.startswith('conv') else (1,0)
            print('number of output channels: {}'.format(num_chan))
            sub_layers = [
                Conv2dMaskable(in_channel, num_chan, kernel_size=k, padding=pad),
                nn.BatchNorm2d(num_chan),
                nn.ReLU(inplace=True)
            ]
            if drop_p is not None:
                sub_layers += [dropout(p=drop_p)]
            layers[name] = nn.Sequential(*sub_layers)
            in_channel = num_chan
        elif name.startswith('pool'):
            k, s = info
            layers[name] = nn.MaxPool2d(kernel_size=k, stride=s)
        elif name.startswith('drop'):
            _, drop_p = info
            layers[name] = dropout(p=drop_p)
        elif name.startswith('logit'):
            layers[name] = Conv2dMaskable(in_channel, num_class, kernel_size=1, padding=0)
        elif name.startswith('flatter'):
            layers[name] = Flatten()
        else:
            assert False
    model = nn.Sequential(layers)
    model.model_conf = model_conf # capture the model_conf for serialization
    init_weights(model, use_in_channel=True)
    model = wrap_model(model)
    return model

def make_data(train):
    transforms = [torchvision.transforms.RandomHorizontalFlip()] if train else [] 
    transforms += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.469, 0.481, 0.451], std=[0.239,0.245,0.272])
    ]
    return torchvision.datasets.CIFAR10(
            '/home/noid/data/torchvision_data/cifar10', 
            train=train, transform=torchvision.transforms.Compose(transforms))

def make_all(batch_size=64, algo='SGD'):
    global dataset, loader, dataset_test, loader_test, model, criterion, optimizer, lr, wd
    dataset      = make_data(train=True)
    dataset_test = make_data(train=False)
    loader       = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_test   = torch.utils.data.DataLoader(
                        dataset_test, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    model = make_model().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    wd = 0.0005
    if algo=='SGD':
        lr = 0.1
        optimizer = MonitoredSGD(model.parameters(), lr, momentum=0.9, weight_decay=wd)
    elif algo=='RMSprop':
        lr = 0.001
        optimizer = MonitoredRMSprop(model.parameters(), lr, weight_decay=wd)
    elif algo=='Adam':
        lr = 0.001
        optimizer = MonitoredAdam(model.parameters(), lr, weight_decay=wd)
    elif algo=='AdamSep':
        lr = 0.001
        optimizer = MonitoredAdam(model.parameters(), lr, weight_decay=0.10, separate_decay=True)
    elif algo=='AdamEps':
        lr = 0.001
        optimizer = MonitoredAdam(model.parameters(), lr, weight_decay=wd, eps=3e-4)
    elif algo=='OldAdam':
        lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=wd)
        optimizer.update_norm = 0
    else:
        assert False

min_loss = 999999
min_loss_batches = 0

def train(num_epoch, report_interval=100, do_validation=True, do_util=False):
    global min_loss, min_loss_batches
    # gc.collect()
    # tracker = SummaryTracker()
    for epoch in xrange(num_epoch):
        gloss = 0
        model.train()
        num_correct = 0
        num_samples = 0
        for batch, (imgs, labels) in enumerate(loader):
            """ forward """
            v_imgs   = Variable(imgs).cuda()
            v_labels = Variable(labels).cuda()
            v_output = model(v_imgs)
            v_loss   = criterion(v_output, v_labels)
            """ backward """
            optimizer.zero_grad()
            v_loss.backward()
            optimizer.step()
            """ monitor progress """
            num_correct += get_num_correct(v_output.data.cpu(), labels)
            num_samples += labels.size(0)
            gloss += v_loss.data[0]
            min_loss_batches += 1
            if (batch+1) % report_interval == 0:
                avg_loss = gloss / report_interval
                gloss = 0
                if avg_loss < min_loss:
                    min_loss = avg_loss
                    min_loss_batches = 0
                print('{:4d}, {:4d}: {:6f} [{:6f}/{:6f}] [{:4d}/{:6f}]'.format(
                    epoch, batch, avg_loss, 
                    get_model_param_norm(model, simple=False), optimizer.update_norm, 
                    min_loss_batches, min_loss))
        print("Epoch {} done. Evaluation:".format(epoch))
        print(num_correct, num_samples)
        if do_validation:
            print(eval_accuracy(loader_test))
        if do_util:
            pprint(get_all_utilization(model))
        # gc.collect()
        # tracker.print_diff()


def eval_accuracy(loader):
    model.eval()
    num_correct = 0
    num_samples = 0
    for imgs, labels in loader:
        v_imgs   = Variable(imgs).cuda()
        v_output = model(v_imgs)
        output = v_output.data.cpu()
        num_correct += get_num_correct(output, labels)
        num_samples += labels.size(0)
    return num_correct, num_samples


class Conv2dMaskable(nn.Conv2d):
    def forward(self, x):
        return nn.Conv2d.forward(self, x)
    def analyze(self):
        return get_layer_utlization(self)

def decayed_training(schedule):
    global lr
    for epochs in schedule:
        train(epochs, 50)
        lr /= 2
        set_learning_rate(optimizer, lr)

