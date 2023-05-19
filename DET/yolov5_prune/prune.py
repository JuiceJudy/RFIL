# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
import yaml
import math

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
sys.path.append('../') 
logger = logging.getLogger(__name__)

# from models.common import *
from models.common_prune import *
import models.common_prune as cp
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr, intersect_dicts, is_parallel
from prune_methods import *

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, prune = False):  # model, input channels, number of classes #prune 是否生成一个压缩模型
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], prune = prune)  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])


        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            # print(m)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch, prune=False):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    prune_rate = d['prune_rate']
    # print(prune_rate)

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        if m not in ['Detect'] : m = 'cp.'+m
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        # print(i,prune_rate[i],m)

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [cp.Conv, GhostConv, cp.Bottleneck, GhostBottleneck, cp.SPP, cp.DWConv, MixConv2d, cp.Focus, CrossConv, cp.BottleneckCSP,
                 cp.C3, cp.C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [cp.BottleneckCSP, cp.C3, cp.C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is cp.Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is cp.Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is cp.Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        # print('---+-->',m)

        # m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        if prune: #prune 是否生成一个压缩模型
            if m is cp.Conv: # 只修剪 Conv, C3
                m_ = nn.Sequential(*[m(*args,rate=prune_rate[i]) for _ in range(n)]) if n > 1 else m(*args,rate=prune_rate[i])
                c2 = int(c2*(1.-prune_rate[i][0]))
            elif m is cp.C3:
                m_ = nn.Sequential(*[m(*args,rate=prune_rate[i]) for _ in range(n)]) if n > 1 else m(*args,rate=prune_rate[i])
            else:
                m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        else :
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def _prunecfg(idx,num=None,rate=[0.],pidx=None,nidx=None,rank=None):
    _cfg = []
    _cfg.append(('idx',idx))
    _cfg.append(('num',num)) # 剩余通道数量
    _cfg.append(('rate',rate))
    _cfg.append(('pidx',pidx)) # next
    _cfg.append(('nidx',nidx))  # pidx  前一层的的id
    _cfg.append(('rank',rank))  # rank
    _cfg = dict(_cfg)
    return _cfg

def get_Convidx(model,yaml):
    layers = yaml['backbone'] + yaml['head']
    prune_rate = yaml['prune_rate']
    pr = prune_rate
    depth_multiple = yaml['depth_multiple']
    width_multiple = yaml['width_multiple']
    backbone_depth = len(yaml['backbone'])

    state_dict = model.state_dict()
    convidx  = []
    # pruneidx = []  # 计划要修剪的层
    prunecfg = []  # 配置
    for k,name in enumerate(state_dict):
        # print(k,name,state_dict[name].size(),len(state_dict[name].size()))
        if len(state_dict[name].size()) == 4:
            convidx.append(k)

    # print(convidx)
    idx = 0
    for i, (f, n, m, args) in enumerate(layers):  # from, number, module, args

        num = None
        if m in ['Conv','C3']:
            num = math.ceil(args[0] * width_multiple)

        if m == 'Conv':
            _num = int(num*(1.-pr[i][0]))
            prunecfg.append((idx,_prunecfg(idx,_num,pr[i],-1,-1))) # -1 表示无意义
            idx += 6
        elif m == 'C3':
            if i < backbone_depth: # head 中的conv暂时不剪
                tmp_pindx = idx-6
            else :
                tmp_pindx = -1
            prunecfg.append((idx,_prunecfg(idx,int(num*width_multiple),[0.],tmp_pindx,-1)))  # 0.5  todo
            prunecfg.append((idx+6,_prunecfg(idx+6,int(num*width_multiple),[0.],tmp_pindx,-1)))
            prunecfg.append((idx+6*2,None))
            idx += 6 * 3
            n = math.ceil(n*depth_multiple)
            for j in range(n):
                _num = int(num*width_multiple*(1.-pr[i][0]))
                # print(idx,num,pr[i][0],_num)
                prunecfg.append((idx,_prunecfg(idx,_num,pr[i],-1,idx+6)))
                prunecfg.append((idx+6,_prunecfg(idx+6,int(num*width_multiple),[0.],idx,-1)))
                idx += 6 * 2        
        elif m == 'Focus':
            prunecfg.append((idx,None))
            idx += 6 
        elif m == 'SPP':
            prunecfg.append((idx,None))
            prunecfg.append((idx+6,None))
            idx += 6 * 2 
        else :
            pass
    # for x in prunecfg:
    #     print(x)
    prunecfg = dict(prunecfg)
    return convidx,prunecfg

def prune_yolov5s(model,cfg,pruner):
    convidx,prunecfg = get_Convidx(model,yaml)
    state_dict = model.state_dict()
    last = len(state_dict) - 6 # 不包括anchor
    for k,name in enumerate(state_dict):
        if k < last and k in convidx:
            # print(k,name)
            if prunecfg[k] is not None:
                rate = prunecfg[k]['rate']
                nidx = prunecfg[k]['nidx']
                if rate[0] > 0.:
                    # rank = RFIL(model,k,rate[0],n = nidx)
                    rank = eval(pruner)(model,k,rate[0],n = nidx)
                else :
                    rank = list(range(len(state_dict[name])))
                prunecfg[k]['rank'] = rank
            # print(k,prunecfg[k])
    return [convidx,prunecfg]
    

def get_rank_byidx(i,prunecfg):
    cfg = prunecfg[i]
    if cfg is None: return -1  # -1 表示前层和本层都未修剪
    num  = cfg['num'] 
    rate = cfg['rate'] 
    rank = cfg['rank']
    if rate[0] <= 0.:
        return list(range(num))
    else:
        return rank

def get_prank_byidx(i,prunecfg):
    cfg = prunecfg[i]
    if cfg is None: return -1
    pidx = cfg['pidx']  
    # print(pidx)
    if pidx != -1:
        return get_rank_byidx(pidx,prunecfg)
    else:
        return -1

def load_pruned_weights(model, old_state_dict, yaml, pcfg=None): # pcfg 剪纸相关配置

    state_dict = model.state_dict()

    # convidx,prunecfg = get_Convidx(model,yaml)
    if pcfg is None:
        convidx,prunecfg = get_Convidx(model,yaml)
    else :
        convidx,prunecfg = pcfg[0],pcfg[1]
    last = len(old_state_dict) - 6 # 不包括anchor
    lastrank = None


    for k,name in enumerate(old_state_dict):

        # print(k,name)

        if k < last and 'anchor' not in name:
            if k in convidx:
                rank = get_rank_byidx(k,prunecfg)
                
                # if rank is None:
                #     rank = list(range(len(state_dict[name])))

                # print(k,len(rank),rank)
                if rank == -1:
                    state_dict[name] = old_state_dict[name]
                else :
                    prank = get_prank_byidx(k,prunecfg)
                    # print(k,rank,prank)
                    if prank == -1:
                        for _i,i in enumerate(rank):
                            # print('->',_i,i)
                            state_dict[name][_i] = old_state_dict[name][i]
                    else :
                        for _i,i in enumerate(rank):
                            for _j,j in enumerate(prank):
                                state_dict[name][_i][_j] = old_state_dict[name][i][j]
                lastrank = rank
            else :
                if k%6 == 5:
                    state_dict[name] = old_state_dict[name]
                else :
                    if lastrank is None or lastrank==-1:
                        state_dict[name] = old_state_dict[name]
                    else :
                        for _i,i in enumerate(lastrank):
                            state_dict[name][_i] = old_state_dict[name][i]
        else :
            state_dict[name] = old_state_dict[name]
    model.load_state_dict(state_dict, strict=False)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img_size', type=int, default=640, help='image size')
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--prune_method', type=str, default='RFIL', help='prune method')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)
    # import yaml  # for torch hub
    yaml_file = Path(opt.cfg).name
    with open(opt.cfg) as f:
        yaml = yaml.load(f, Loader=yaml.SafeLoader)

    # Create model
    model = Model(opt.cfg).to(device)

    ckpt = torch.load(opt.weights, map_location=device)
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    exclude = ['anchor']
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect

    model.load_state_dict(state_dict, strict=False)  # load

    pcfg = prune_yolov5s(model,yaml,opt.prune_method) 

    pruned_model = Model(opt.cfg, prune = True).to(device)

    # print(pruned_model)
    logger.info(pruned_model)
    model_info(pruned_model, verbose=False, img_size=opt.img_size)

    old_state_dict = model.state_dict()
    pruned_model = load_pruned_weights(pruned_model,old_state_dict,yaml,pcfg)

   
    ckpt = {'epoch': -1,
            'best_fitness': None,
            'training_results': None,
            'model': deepcopy(pruned_model.module if is_parallel(pruned_model) else pruned_model).half(),
            'optimizer': None,
            }
    save_dir = 'prune_runs/pruned.pt'
    torch.save(ckpt, save_dir)
    logger.info('save prune_runs/pruned.pt')

