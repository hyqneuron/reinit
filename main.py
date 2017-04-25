from huva.th_util import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision
from collections import OrderedDict
from pprint import pprint
import gc
import argparse
from pympler.tracker import SummaryTracker
from multi_gpu import wrap_model
import os

"""
TODOs:
1. Figure out ordinary dropout
2. figure out (increase eps, separate weight decay) actually improve things
3. (remove 0-unit, remove low-quantile unit)
"""

parser = argparse.ArgumentParser()
parser.add_argument('--name',       type=str, default='')
parser.add_argument('--optimizer',  type=str, default='SGD')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--thread-train', type=int, default=4)
parser.add_argument('--thread-test',  type=int, default=2)
parser.add_argument('--gpus', type=lambda xs:map(int, xs.split(',')), default=[0])
args = parser.parse_args()

pprint(args)

model_conf_11 = [
    ('input',   (3,   None)),
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
    ('conv5_1', (512, None)),
    ('conv5_2', (512, None)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (512, None)),
    ('logit'  , (10,  None)),
    ('flatter', (None,None))
]

model_conf_16 = [
    ('input',   (3,   None)),
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
    ('logit'  , (10,  None)),
    ('flatter', (None,None))
]

model_conf = model_conf_11

def make_model(activation=nn.ReLU):
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
                activation(inplace=True)
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
            num_class, _ = info
            layers[name] = Conv2dMaskable(in_channel, num_class, kernel_size=1, padding=0)
        elif name.startswith('flatter'):
            layers[name] = Flatten()
        else:
            assert False
    model = nn.Sequential(layers)
    model.model_conf = model_conf # capture the model_conf for serialization
    init_weights(model)
    model = wrap_model(model, args)
    return model

def make_data(batch_size):
    global dataset, dataset_test, loader, loader_test
    (dataset, loader), (dataset_test, loader_test) = make_data_cifar10(batch_size)

def make_all(name, batch_size=64, algo='SGD', ignore_model=False, activation=nn.ReLU):
    global model, criterion, optimizer, lr, wd, logger
    global model_path, log_path
    make_data(batch_size)
    if not ignore_model:
        model = make_model(activation=activation).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    wd = 0.0005
    if algo=='SGD':
        lr = 0.1
        optimizer = MonitoredSGD(model.parameters(), lr, momentum=0.9, weight_decay=wd)
    elif algo=='SGD_fc6wd':
        lr = 0.1
        params = {id(p):p for p in model.parameters()}
        fc6ps  = list(model.fc6.parameters())
        for p in fc6ps: del params[id(p)]
        params = params.values()
        param_groups = [
                {'params':params,'weight_decay':wd}, 
                {'params':fc6ps, 'weight_decay':wd*0.5}
        ]
        optimizer = MonitoredSGD(param_groups, lr, momentum=0.9, weight_decay=wd)
    elif algo=='SGD_conv5wd':
        lr = 0.1
        params = {id(p):p for p in model.parameters()}
        conv5ps = sum(map(list, [model.conv5_1.parameters(), model.conv5_2.parameters(), model.conv5_3.parameters()]),[])
        for p in conv5ps: del params[id(p)]
        params = params.values()
        param_groups = [
                {'params': params, 'weight_decay':wd},
                {'params': conv5ps,'weight_decay':wd*2} # *2 for model30, *4 for model31
        ]
        optimizer = MonitoredSGD(param_groups, lr, momentum=0.9, weight_decay=wd)
    elif algo=='SGD_endwd':
        lr = 0.1
        params = {id(p):p for p in model.parameters()}
        endps = sum(map(lambda l:list(l.parameters()), [model.conv5_1, model.conv5_2, model.conv5_3, model.fc6]),[])
        for p in endps: del params[id(p)]
        params = params.values()
        param_groups = [
                {'params': params, 'weight_decay':wd},
                {'params': endps , 'weight_decay':wd*2}
        ]
        optimizer = MonitoredSGD(param_groups, lr, momentum=0.9, weight_decay=wd)
    elif algo=='SGD_logit':
        lr = 0.1
        params  = {id(p):p for p in model.parameters()}
        logitps = list(model.logit.parameters())
        for p in logitps: del params[id(p)]
        params = params.values()
        param_groups = [
                {'params': params, 'weight_decay':wd},
                {'params': logitps,'weight_decay':wd*2} # *2 for model30, *4 for model31
        ]
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
    model_path = 'logs/{}.pth'.format(name)
    log_path   = 'logs/{}.log'.format(name)
    assert not os.path.exists(model_path)
    assert not os.path.exists(log_path)
    logger = LogPrinter(log_path)


min_loss = 999999
min_loss_batches = 0
epoch_trained = 0

def train(num_epoch, report_interval=100, do_validation=True, do_util=False):
    global min_loss, min_loss_batches, epoch_trained
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
                logger.log('{:3d},{:3d},{:4d}: {:6f} [{:6f}/{:6f}] [{:4d}/{:6f}]'.format(
                    epoch_trained, epoch, batch, avg_loss, 
                    get_model_param_norm(model), optimizer.update_norm, 
                    min_loss_batches, min_loss))
        logger.log("Epoch {} done. Evaluation:".format(epoch))
        logger.log((num_correct, num_samples))
        if do_validation:
            logger.log(eval_accuracy(loader_test))
        if do_util:
            pprint(get_model_utilization(model))
    epoch_trained += 1


def eval_accuracy(loader, testmodel=None, max_batches=99999999):
    if testmodel is None: testmodel = model
    testmodel.eval()
    num_correct = 0
    num_samples = 0
    for i, (imgs, labels) in enumerate(loader):
        if i >= max_batches:
            break
        v_imgs   = Variable(imgs).cuda()
        v_output = testmodel(v_imgs)
        output = v_output.data.cpu()
        num_correct += get_num_correct(output, labels)
        num_samples += labels.size(0)
    return num_correct, num_samples

def decayed_training(schedule):
    global lr
    for epochs in schedule:
        train(epochs, 50)
        lr /= 2
        set_learning_rate(optimizer, lr)
    model.logtxt = logger.logtxt
    torch.save(model, model_path)

def make_model_keep_output(model, value=True):
    for mod in model.modules():
        if type(mod) in [BNMaskable, Conv2dMaskable, nn.BatchNorm2d]:
            mod.keep_output = value

def eval_1sample(model):
    model.eval()
    make_model_keep_output(model, True)
    img0, label0 = dataset_test[0]
    v_imgs = Variable(img0.unsqueeze(0)).cuda()
    v_output = model(v_imgs)
    output0 = v_output.data[0]
    return output0

class Conv2dMaskable(nn.Conv2d):
    def forward(self, x):
        out = nn.Conv2d.forward(self, x)
        if hasattr(self, 'alive_mask'):
            out =  out * Variable(self.alive_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(out))
        else:
            out = nn.Conv2d.forward(self, x)
        if hasattr(self, 'keep_output') and self.keep_output:
            self.output = out
        return out
    def analyze(self):
        return get_layer_utilization(self)
    def set_prev_mask(self, threshold=1e-10):
        if self.prev_conv is None: return
        summary = self.analyze()
        self.prev_conv.alive_mask = summary.gt(threshold).float().cuda()
    def output_selector(self):
        if not hasattr(self, 'alive_mask'):
            return torch.range(0, self.out_channels-1).long().cuda()
        num_selected = int(self.alive_mask.sum())
        selector = torch.LongTensor(num_selected)
        idx = 0
        for i in range(self.out_channels):
            if self.alive_mask[i] > 0:
                selector[idx] = i
                idx += 1
        assert idx == selector.size(0)
        return selector.cuda()
    def input_selector(self):
        if self.prev_conv is None:
            return torch.range(0, self.in_channels-1).long().cuda()
        else:
            return self.prev_conv.output_selector()

class BNMaskable(nn.BatchNorm2d):
    def forward(self, x):
        out = nn.BatchNorm2d.forward(self, x)
        if hasattr(self, 'keep_output') and self.keep_output:
            self.output = out
        return out


def chainup_convs(model):
    """
    Chain up conv layers so that a subsequent layer can decide its previous layer's units 'aliveness'
    """
    prev_conv = None
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            mod.prev_conv = prev_conv
            assert isinstance(mod, Conv2dMaskable)
            mod.set_prev_mask()
            prev_conv = mod
        elif isinstance(mod, nn.BatchNorm2d):
            mod.prev_conv = prev_conv

def copy_conv2d_without_dead(conv2d):
    assert isinstance(conv2d, Conv2dMaskable)
    weight_selector = conv2d.output_selector()
    num_selected = weight_selector.size(0)
    input_selector = conv2d.input_selector()
    new_weight = nn.Parameter(conv2d.weight.data[weight_selector].index_select(1, input_selector).clone().cuda())
    new_bias   = nn.Parameter(conv2d.bias.data  [weight_selector].clone())
    in_channels = input_selector.size(0)
    new_conv2d = Conv2dMaskable(in_channels, num_selected, 
                                kernel_size=conv2d.kernel_size, padding=conv2d.padding)
    new_conv2d.weight = new_weight
    new_conv2d.bias   = new_bias
    return new_conv2d

def copy_bn_without_dead(conv2d, bn):
    assert conv2d.out_channels==bn.num_features
    assert isinstance(conv2d, Conv2dMaskable)
    weight_selector = conv2d.output_selector()
    num_selected = weight_selector.size(0)
    new_weight = nn.Parameter(bn.weight.data[weight_selector].clone())
    new_bias   = nn.Parameter(bn.bias.data  [weight_selector].clone())
    new_bn = BNMaskable(num_selected)
    new_bn.weight = new_weight
    new_bn.bias   = new_bias
    new_bn.running_mean.copy_(bn.running_mean[weight_selector])
    new_bn.running_var.copy_ (bn.running_var[weight_selector])
    bn.__class__ = BNMaskable
    return new_bn

def copy_sequential_without_dead(sequential):
    """
    copy the entire model, when module is conv2d, only copy non-dead units
    """
    assert isinstance(sequential, nn.Sequential)
    mods = OrderedDict()
    prev_conv = None
    for name, mod in sequential.named_children():
        if isinstance(mod, nn.Conv2d):
            prev_conv = mod
            mod = copy_conv2d_without_dead(mod)
        elif isinstance(mod, nn.BatchNorm2d) or isinstance(mod, BNMaskable):
            assert prev_conv is not None
            mod = copy_bn_without_dead(prev_conv, mod)
        elif isinstance(mod, nn.Sequential):
            mod = copy_sequential_without_dead(mod)
        elif isinstance(mod, nn.MaxPool2d):
            mod = nn.MaxPool2d(kernel_size=mod.kernel_size, stride=mod.stride).cuda()
        elif isinstance(mod, nn.Dropout) or isinstance(mod, nn.Dropout2d):
            mod = mod.__class__(mod.p).cuda()
        elif isinstance(mod, nn.ReLU):
            mod = nn.ReLU(True).cuda()
        elif isinstance(mod, Flatten):
            mod = Flatten()
        else:
            assert False, 'unknown module type {}'.format(mod.__class__)
        mods[name] = mod
    return nn.Sequential(mods).cuda()

import matplotlib.pyplot as plt
def vis_output_hist(outputt, output_std, i, bins=40, mode='show', savename=None):
    plt.hist(outputt[i].numpy(), bins=bins)
    if mode == 'show':
        print(output_std[i])
        plt.show()
    elif mode == 'save':
        assert savename is not None
        plt.savefig(savename)
        plt.close()
    else:
        assert False

def vis_hist(i, bins=40):
    vis_output_hist(output_stats.outputt, output_stats.std, i, bins=bins)

def compute_stats(name, layer):
    global output, stats
    name_output = collect_output_over_loader(model, {name:layer}, loader_test, max_batches=40)
    output = name_output[name]
    stats  = get_output_stats(output)
    return output, stats

def save_hist_for_layer(name, layer, save_prefix=''):
    output, stats = compute_stats()
    """ save by normal order """
    for i in xrange(stats.outputt.size(0)):
        save_name = save_prefix + '{}_{}_{:4f}_{:4f}.jpg'.format(name, i, stats.std[i], stats.skew[i])
        save_output_hist(stats.outputt, i, save_name)


def save_hist_for_all_convs(save_prefix=''):
    name_layer = {name:layer for name,layer in model.named_modules() if isinstance(layer, Conv2dMaskable)}
    name_output = collect_output_over_loader(model, name_layer, loader_test, max_batches=20)
    name_stats = {name:get_output_stats(output) for name,output in name_output.iteritems()}
    for name, stats in name_stats.iteritems():
        print("writing histograms for {}".format(name))
        num_units = stats.outputt.size(0)
        """ save by order of std """
        ordered_indices = torch.range(0, num_units-1)[stats.order_std]
        for i in xrange(num_units):
            actual_index = ordered_indices[i]
            savename = save_prefix + '{}_{}[{}].jpg'.format(name, i, stats.std[actual_index])
            vis_output_hist(stats.outputt, stats.std, actual_index, mode='save', savename=savename)

def t1():
    """
    Copy without dead units
    """
    global model, model2
    model = torch.load('logs/model23.pth')
    make_data(128)
    print(eval_accuracy(loader_test))
    chainup_convs(model)
    print(eval_accuracy(loader_test))
    model2 = copy_sequential_without_dead(model)
    eval_1sample(model)
    eval_1sample(model2)

def t2():
    """
    Compute stats on a particular layer
    """
    global model, layer, stats
    model = torch.load('logs/model29.pth')
    make_data(128)
    layer = model.conv5_1[0]
    name_layer = {'conv5_1':model.conv5_1[0]}
    name_output= collect_output_over_loader(model, name_layer, loader_test)
    stats = get_output_stats(name_output['conv5_1'])

def t3():
    """
    save stats for all conv layers
    """
    global model
    model = torch.load('logs/model39.pth')
    make_data(128)
    save_stats_for_all_convs(model, 'logs/model39_graphs/')

def t4(idx=0, layer=None, backward_thresh=True):
    """
    compute guided backprop, without loading model
    """
    global data, grad
    if layer is None:
        layer = model.conv3_1[2]
    data, grad = collect_output_and_guided_backprop(model, layer, idx, dataset_test, loader_test, 
            max_batches=20, top_k=10, backward_thresh=backward_thresh)

def t5():
    """
    compute guided backprop
    """
    load_model()
    make_data(128)
    t4()

def normalize(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor

def save_layer_gbp(model, layer, save_prefix=''):
    """
    Visualize a layer using guided backprop
    """
    global data, grad
    assert isinstance(layer, nn.Conv2d)
    assert layer in model.modules()
    C = layer.out_channels
    name_output = collect_output_over_loader(model, {0:layer}, loader_test, max_batches=40)
    output = name_output[0]
    for c in xrange(C):
        data, grad = guided_backprop_layer_unit(model, layer, output, c, dataset_test, top_k=10)
        vis_gbp(savepath='{}{}.jpg'.format(save_prefix, c))
    print('done')

def vis_gbp(savepath=None):
    """
    visualize data and grad computed from guided backpropagation
    """
    global data, grad
    data = normalize(data)
    grad = normalize(grad)
    N = data.size(0)
    f, axs = plt.subplots(2, N, figsize=(19,5))
    for i in xrange(N):
        #plt.subplot(2, N, 1+i)
        axs[0,i].imshow(data[i].cpu().numpy().transpose([1,2,0]))
        axs[0,i].axis('off')
        #plt.subplot(2, N, 1+i+N)
        axs[1,i].imshow(grad[i].cpu().numpy().transpose([1,2,0]))
        axs[1,i].axis('off')
    plt.tight_layout()
    if savepath is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(savepath)
        plt.close()

def load_model(path='logs/model34.pth'):
    global model
    model = torch.load(path)

if __name__=='__main__':
    if args.name != '':
        make_all(args.name, args.batch_size, args.optimizer)
        decayed_training([30]*10)
