# *_*coding:utf-8 *_*
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import logging
import pandas as pd
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def show_example(x, y, x_reconstruction, y_pred,save_dir, figname):
    x = x.squeeze().cpu().data.numpy()
    x = x.permute(0,2,1)
    y = y.cpu().data.numpy()
    x_reconstruction = x_reconstruction.squeeze().cpu().data.numpy()
    _, y_pred = torch.max(y_pred, -1)
    y_pred = y_pred.cpu().data.numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x, cmap='Greys')
    ax[0].set_title('Input: %d' % y)
    ax[1].imshow(x_reconstruction, cmap='Greys')
    ax[1].set_title('Output: %d' % y_pred)
    plt.savefig(save_dir + figname + '.png')

def save_pruned_checkpoint(epoch, train_accuracy, test_accuracy, model, optimizer, pruning_rate, path,modelnet='checkpoint'):
    savepath  = path + '/%s-%f-%04d.pth' % (modelnet,test_accuracy, epoch)
    state = {
        'epoch': epoch,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'pruning_rate':pruning_rate,
    }
    torch.save(state, savepath)

def save_checkpoint(epoch, train_accuracy, test_accuracy, model, optimizer, path,modelnet='checkpoint'):
    savepath  = path + '/%s-%f-%04d.pth' % (modelnet,test_accuracy, epoch)
    state = {
        'epoch': epoch,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)

def test(model, loader):
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    return accuracy

def compute_cat_iou(pred,target,iou_tabel):
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def test_partseg(model, loader, catdict, num_classes = 50,forpointnet2=False):
    ''' catdict = {0:Airplane, 1:Airplane, ...49:Table} '''
    iou_tabel = np.zeros((len(catdict),3))
    iou_list = []
    metrics = defaultdict(lambda:list())
    hist_acc = []
    # mean_correct = []
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point,_= points.size()
        points, label, target, norm_plt = Variable(points.float()),Variable(label.long()), Variable(target.long()),Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(), label.squeeze().cuda(), target.cuda(), norm_plt.cuda()
        if forpointnet2:
            seg_pred = model(points, norm_plt, to_categorical(label, 16))
        else:
            labels_pred, seg_pred, _  = model(points,to_categorical(label,16))
            # labels_pred_choice = labels_pred.data.max(1)[1]
            # labels_correct = labels_pred_choice.eq(label.long().data).cpu().sum()
            # mean_correct.append(labels_correct.item() / float(points.size()[0]))
        # print(pred.size())
        iou_tabel, iou = compute_cat_iou(seg_pred,target,iou_tabel)
        iou_list+=iou
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        seg_pred = seg_pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['inctance_avg_iou'] = np.mean(iou_list)
    # metrics['label_accuracy'] = np.mean(mean_correct)
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    metrics['class_avg_iou'] = np.mean(cat_iou)

    return metrics, hist_acc, cat_iou

def test_semseg(model, loader, catdict, num_classes = 13, pointnet2=False):
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        if pointnet2:
            pred = model(points[:, :3, :], points[:, 3:, :])
        else:
            pred, _ = model(points)
        # print(pred.size())
        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    # print(iou_tabel)
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou


def compute_avg_curve(y, n_points_avg):
    avg_kernel = np.ones((n_points_avg,)) / n_points_avg
    rolling_mean = np.convolve(y, avg_kernel, mode='valid')
    return rolling_mean

def plot_loss_curve(history,n_points_avg,n_points_plot,save_dir):
    curve = np.asarray(history['loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-g')

    curve = np.asarray(history['margin_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-b')

    curve = np.asarray(history['reconstruction_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-r')

    plt.legend(['Total Loss', 'Margin Loss', 'Reconstruction Loss'])
    plt.savefig(save_dir + '/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M')) + '_total_result.png')
    plt.close()

def plot_acc_curve(total_train_acc,total_test_acc,save_dir):
    plt.plot(total_train_acc, '-b',label = 'train_acc')
    plt.plot(total_test_acc, '-r',label = 'test_acc')
    plt.legend()
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('Accuracy of training and test')
    plt.savefig(save_dir +'/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'_total_acc.png')
    plt.close()

def show_point_cloud(tuple,seg_label=[],title=None):
    import matplotlib.pyplot as plt
    if seg_label == []:
        x = [x[0] for x in tuple]
        y = [y[1] for y in tuple]
        z = [z[2] for z in tuple]
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    else:
        category = list(np.unique(seg_label))
        color = ['b','r','g','y','w','b','p']
        ax = plt.subplot(111, projection='3d')
        for categ_index in range(len(category)):
            tuple_seg = tuple[seg_label == category[categ_index]]
            x = [x[0] for x in tuple_seg]
            y = [y[1] for y in tuple_seg]
            z = [z[2] for z in tuple_seg]
            ax.scatter(x, y, z, c=color[categ_index], cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    plt.title(title)
    plt.show()

# yzhid 2023.04.19
def model_info(model, verbose=False, npoints=1024):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        input = torch.zeros((1, 3, 100), device=next(model.parameters()).device) 
        flops = profile(deepcopy(model), inputs=(input,input), verbose=False)[0] 
        fs = ', %.1f FLOPs' % flops
    except (ImportError, Exception):
        fs = ''

    print('Model Summary: {} layers, {} parameters, {} gradients{}'.format(len(list(model.modules())),n_p,n_g,fs))
    return flops, n_p

def format_compress_rate(compress_rate):
    if compress_rate is None : return compress_rate
    import re
    cprate_str=compress_rate
    cprate_str_list=cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate=[]
    for x in cprate_str_list:
        num=1
        find_num=re.findall(pat_num,x)
        if find_num:
            assert len(find_num) == 1
            num=int(find_num[0].replace('*',''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate)>=1
        cprate+=[float(x) for x in find_cprate]*num
    return cprate

def get_logger(filename, verbosity=1, name=None):

    time_format = "%Y-%m-%d %H:%M:%S"
    log_dir = Path('logs/')
    log_dir.mkdir(exist_ok=True)

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",time_format
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def all_layers(model):
    layers = []
    _name_ = ['weight','bias','running_mean','running_var','num_batches_tracked']
    state_dict = model.state_dict()
    for k,name in enumerate(state_dict):
        # print(k,name,state_dict[name].size())
        if any(x in name for x in _name_):
            layers.append(name)
    # print(len(layers),layers)
    return layers
