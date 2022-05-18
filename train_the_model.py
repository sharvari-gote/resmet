import os
import sys
import pickle
import json
import numpy as np
import argparse
from collections import OrderedDict
import torch.nn.functional as F
import functional as LF
import torch.nn as nn

import inspect
src_file_path = inspect.getfile(lambda: None)

import torch
from torchvision import transforms
from torch.utils.model_zoo import load_url

from datasets import *
from backbone import *
from SiameseNet import *
from augmentations import augmentation
from utils import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#from code.utils.datasets import *
#from code.utils.utils import *
#from code.networks.backbone import *
#from code.networks.SiameseNet import *
#from code.utils.augmentations import augmentation


'''Script for the extraction of descriptors for the Met dataset given a (pretrained) backbone.
'''
class MAC(nn.Module):

    def __init__(self):
        super(MAC,self).__init__()

    def forward(self, x):
        return LF.mac(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC,self).__init__()

    def forward(self, x):
        return LF.spoc(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class GeMmp(nn.Module):

    def __init__(self, p=3, mp=1, eps=1e-6):
        super(GeMmp,self).__init__()
        self.p = Parameter(torch.ones(mp)*p)
        self.mp = mp
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p.unsqueeze(-1).unsqueeze(-1), eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '[{}]'.format(self.mp) + ', ' + 'eps=' + str(self.eps) + ')'

class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6):
        super(RMAC,self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'

# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    'vgg16'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

# TODO: pre-compute for more architectures and properly test variations (pre l2norm, post l2norm)
# pre-computed local pca whitening that can be applied before the pooling layer
L_WHITENING = {
    'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-9f830ef.pth', # no pre l2 norm
    # 'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-da5c935.pth', # with pre l2 norm
}

# possible global pooling layers, each on of these can be made regional
POOLING = {
    'mac'   : MAC,
    'spoc'  : SPoC,
    'gem'   : GeM,
    'gemmp' : GeMmp,
    'rmac'  : RMAC,
}

# TODO: pre-compute for: resnet50-gem-r, resnet50-mac-r, vgg16-mac-r, alexnet-mac-r
# pre-computed regional whitening, for most commonly used architectures and pooling methods
R_WHITENING = {
    'alexnet-gem-r'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-rwhiten-c8cf7e2.pth',
    'vgg16-gem-r'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-rwhiten-19b204e.pth',
    'resnet101-mac-r' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-rwhiten-7f1ed8c.pth',
    'resnet101-gem-r' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-rwhiten-adace84.pth',
}

# TODO: pre-compute for more architectures
# pre-computed final (global) whitening, for most commonly used architectures and pooling methods
WHITENING = {
    'alexnet-gem'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth',
    'alexnet-gem-r'          : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-whiten-4c9126b.pth',
    'vgg16-gem'              : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth',
    'vgg16-gem-r'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-whiten-83582df.pth',
    'resnet50-gem'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet50-gem-whiten-f15da7b.pth',
    'resnet101-mac-r'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-whiten-9df41d3.pth',
    'resnet101-gem'          : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
    'resnet101-gem-r'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-whiten-b379c0a.pth',
    'resnet101-gemmp'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gemmp-whiten-770f53c.pth',
    'resnet152-gem'          : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet152-gem-whiten-abe7b93.pth',
    'densenet121-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet121-gem-whiten-79e3eea.pth',
    'densenet169-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet169-gem-whiten-6b2a76a.pth',
    'densenet201-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet201-gem-whiten-22ea45c.pth',
}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet'               :  256,
    'vgg11'                 :  512,
    'vgg13'                 :  512,
    'vgg16'                 :  512,
    'vgg19'                 :  512,
    'resnet18'              :  512,
    'resnet34'              :  512,
    'resnet50'              : 2048,
    'resnet101'             : 2048,
    'resnet152'             : 2048,
    'densenet121'           : 1024,
    'densenet169'           : 1664,
    'densenet201'           : 1920,
    'densenet161'           : 2208, # largest densenet
    'squeezenet1_0'         :  512,
    'squeezenet1_1'         :  512,
}

def get_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(src_file_path)))))


def get_data_root():
    return os.path.join(get_root(), 'data')

class Rpool(nn.Module):

    def __init__(self, rpool, whiten=None, L=3, eps=1e-6):
        super(Rpool,self).__init__()
        self.rpool = rpool
        self.L = L
        self.whiten = whiten
        self.norm = L2N()
        self.eps = eps

    def forward(self, x, aggregate=True):
        # features -> roipool
        o = LF.roipool(x, self.rpool, self.L, self.eps) # size: #im, #reg, D, 1, 1

        # concatenate regions from all images in the batch
        s = o.size()
        o = o.view(s[0]*s[1], s[2], s[3], s[4]) # size: #im x #reg, D, 1, 1

        # rvecs -> norm
        o = self.norm(o)

        # rvecs -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))

        # reshape back to regions per image
        o = o.view(s[0], s[1], s[2], s[3], s[4]) # size: #im, #reg, D, 1, 1

        # aggregate regions into a single global vector per image
        if aggregate:
            # rvecs -> sumpool -> norm
            o = self.norm(o.sum(1, keepdim=False)) # size: #im, D, 1, 1

        return o

    def __repr__(self):
        return super(Rpool, self).__repr__() + '(' + 'L=' + '{}'.format(self.L) + ')'

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.l2n(x, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0,2,3,1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return 



def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: lwhiten with possible dimensionality reduce

        if pretrained:
            lw = architecture
            if lw in L_WHITENING:
                print(">> {}: for '{}' custom computed local whitening '{}' is used"
                    .format(os.path.basename(src_file_path), lw, os.path.basename(L_WHITENING[lw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                lwhiten.load_state_dict(load_url(L_WHITENING[lw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                    .format(os.path.basename(src_file_path), lw))

    else:
        lwhiten = None
    
    # initialize pooling
    if pooling == 'gemmp':
        pool = POOLING[pooling](mp=dim)
    else:
        pool = POOLING[pooling]()
    
    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            if rw in R_WHITENING:
                print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                    .format(os.path.basename(src_file_path), rw, os.path.basename(R_WHITENING[rw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                rwhiten.load_state_dict(load_url(R_WHITENING[rw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                    .format(os.path.basename(src_file_path), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening
    if whitening:
        whiten = nn.Linear(dim, dim, bias=True)
        # TODO: whiten with possible dimensionality reduce

        if pretrained:
            w = architecture
            if local_whitening:
                w += '-lw'
            w += '-' + pooling
            if regional:
                w += '-r'
            if w in WHITENING:
                print(">> {}: for '{}' custom computed whitening '{}' is used"
                    .format(os.path.basename(src_file_path), w, os.path.basename(WHITENING[w])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                whiten.load_state_dict(load_url(WHITENING[w], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no whitening computed, random weights are used"
                    .format(os.path.basename(src_file_path), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'local_whitening' : local_whitening, 
        'pooling' : pooling, 
        'regional' : regional, 
        'whitening' : whitening, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(src_file_path), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(load_url(FEATURES[architecture], model_dir=model_dir))

    return net



def main():
    '''
    parser = argparse.ArgumentParser() #parser.add_argument('directory', metavar='EXPORT_DIR', default='./ext',help='d')
    parser.add_argument('directory', metavar='EXPORT_DIR', default='./ext',help='d')
    parser.add_argument('EXPORT_DIR', default='./ext', metavar='EXPORT_DIR', help='directory')
    parser.add_argument('--gpuid', default=0, type=int) #id of the gpu in your machine
    parser.add_argument('--net', default='r18INgem')
    parser.add_argument('--netpath', default=None) #optional
    parser.add_argument('--ms', action='store_true') #multiscale descriptors
    parser.add_argument('--mini', action='store_true') #use the mini database
    parser.add_argument('--queries_only', action='store_true')
    parser.add_argument('--trained_on_mini', action='store_true') #if your model has a classification head for the mini dataset
    parser.add_argument('--info_dir',default=None, type=str, help = 'directory where ground truth is stored')
    parser.add_argument('--im_root',default=None, type=str, help = 'directory where images are stored')
    #parser.add_argument('EXPORT_DIR', default='./ext', metavar='EXPORT_DIR', help='directory')
    args = parser.parse_args()
    '''
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	# folder name
    #network_variant = args.net
    #network_variant = 'r18INgem'
    #network_variant = 'r50INgem'
    #network_variant = 'r50_swav_gem'
    #network_variant = 'r50_SIN_gem'
    #network_variant = 'r18_sw-sup_gem'
    #network_variant = 'r50_sw-sup_gem'
    #network_variant = 'resnext50_32x4d_swsl'
    #network_variant = 'resnext101_32x4d_swsl'
    #network_variant = 'resnext101_32x8d_swsl'
    network_variant = 'resnext101_32x16d_swsl'
    miniFlag = True
    mini = None
    
    network_variant = input("Enter the network variant:")
    while(miniFlag):
        miniOrNot = input("Please enter 1 if you are training on the mini dataset. Else, enter 0 for training on the entire dataset")
        if(miniOrNot == "1"):
            mini = True
        elif(miniOrNot == "0"):
            mini = False
        else:
            print("Invalid input. Try again.")
    exp_name = network_variant
    exp_name+=("_ms")
    if mini:
        exp_name+=("_mini")
	
    exp_name+=("_mini")
    
    exp_dir = "./ext"+"/"+exp_name+"/"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
	
    print("Will save descriptors in {}".format(exp_dir))
	
    extraction_transform = augmentation("augment_inference")

    #train_root = args.info_dir
    image_root = './codeFiles/code/datasets_dir/'
    train_root = './codeFiles/code/datasets_dir/ground_truth'
    if(mini):
        num_classes = 33501
        train_dataset = MET_database(root = train_root,mini= True,transform = extraction_transform,im_root = im_root)
    else:
        num_classes = 224408 
        train_dataset = MET_database(root = train_root,transform = extraction_transform,im_root = im_root)
    query_root = train_root
    #train_dataset = MET_database(root = train_root,transform = extraction_transform,im_root = './METFiles/')
    train_dataset = MET_database(root = train_root,mini= True,transform = extraction_transform,im_root = im_root)
     #test_dataset = MET_queries(root = query_root,test = True,transform = extraction_transform,im_root = args.im_root)
    test_dataset = MET_queries(root = query_root,test = True,transform = extraction_transform,im_root = im_root)
    #val_dataset = MET_queries(root = query_root,transform = extraction_transform,im_root = args.im_root)
    val_dataset = MET_queries(root = query_root,transform = extraction_transform,im_root = im_root)
    
    '''
    if not args.queries_only:
	    train_dataset = MET_database(root = train_root,mini = args.mini,transform = extraction_transform,im_root = args.im_root)
	
    
    if args.trained_on_mini:
	    num_classes = 33501

    else:
	    num_classes = 224408
    '''


   
    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print("Number of train images: {}".format(len(train_dataset)))
    '''
    if not args.queries_only:
	    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
	    print("Number of train images: {}".format(len(train_dataset)))
    '''
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print("Number of test images: {}".format(len(test_dataset)))
    print("Number of val images: {}".format(len(val_dataset)))

    netpath = None
	#initialization of the global descriptor extractor model
    #if args.netpath is not None:
    if netpath is not None:
	    if network_variant == 'r18_contr_loss_gem':
		    model = siamese_network("resnet18",pooling = "gem",pretrained = False)
		    print("loading weights from checkpoint")
		    model.load_state_dict(torch.load(args.netpath)['state_dict'])
		    net = model.backbone

	    elif network_variant == 'r18_contr_loss_gem_fc':
		    model = siamese_network("resnet18",pooling = "gem",pretrained = False,
			    emb_proj = True)
		    model.backbone.projector.bias.data = model.backbone.projector.bias.data.unsqueeze(0)
		    print("loading weights from checkpoint")
		    model.load_state_dict(torch.load(args.netpath)['state_dict'])
		    net = model.backbone

	    elif network_variant == 'r18_contr_loss_gem_fc_swsl':
		    model = siamese_network("r18_sw-sup",pooling = "gem",pretrained = False,
			    emb_proj = True)
		    model.backbone.projector.bias.data = model.backbone.projector.bias.data.unsqueeze(0)
		    print("loading weights from checkpoint")
		    model.load_state_dict(torch.load(args.netpath)['state_dict'])
		    net = model.backbone
			
	    else:
		    raise ValueError('Unsupported  architecture: {}!'.format(network_variant))

    else:
		
	    if network_variant == 'r18INgem':
		    net = Embedder("resnet18",gem_p = 3.0,pretrained_flag = True)
		
	    elif network_variant == 'r50INgem_caffe': #we use this version because it has the weights from caffe library which perform better
		    net_params = {'architecture':"resnet50",'pooling':"gem",'pretrained':True,'whitening':False}	
		    net = init_network(net_params)

	    elif network_variant == 'r50INgem': #pytorch weights
		    net = Embedder('resnet50',gem_p = 3.0,pretrained_flag = True,projector = False)

	    elif network_variant == 'r50_swav_gem':
		    model = torch.hub.load('facebookresearch/swav','resnet50')
		    net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
		    net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}

	    elif network_variant == 'r50_SIN_gem':

		    model = torchvision.models.resnet50(pretrained=False)
		    checkpoint =load_url('https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar')
		    model = torch.nn.DataParallel(model)
		    model.load_state_dict(checkpoint['state_dict'])
		    model = model.module
		    net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
		    net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}
		
	    elif network_variant == 'r18_sw-sup_gem':
		    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
		    net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
		    net.meta = {
				'architecture' : "resnet18", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 512,
			}

	    elif network_variant == 'r50_sw-sup_gem':
		    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
		    net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
		    net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}
	    elif network_variant == 'resnext50_32x4d_swsl':
		    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
		    net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
		    net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}

	    elif network_variant == 'resnext101_32x4d_swsl':
		    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
		    net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
		    net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}

	    elif network_variant == 'resnext101_32x8d_swsl':
		    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
		    net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
		    net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}

	    elif network_variant == 'resnext101_32x16d_swsl':
		    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
		    net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
		    net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}
	    else:
		    raise ValueError('Unsupported  architecture: {}!'.format(network_variant))


    net.cuda()
    scales = [1, 1/np.sqrt(2), 1/2]
    '''
    if args.ms:
		#multi-scale case
	    scales = [1, 1/np.sqrt(2), 1/2]

    else:
		#single-scale case
	    scales = [1]
    '''
    
    print("Starting the extraction of the descriptors")
    train_descr = extract_embeddings(net,train_loader,ms = scales,msp = 1.0,print_freq=1000)
    print("Train descriptors finished...")
    test_descr = extract_embeddings(net,test_loader,ms = scales,msp = 1.0,print_freq=1000)
    print("Test descriptors finished...")
    val_descr = extract_embeddings(net,val_loader,ms = scales,msp = 1.0,print_freq=1000)
    print("Val descriptors finished...")
    
    '''
    if not args.queries_only:
	    train_descr = extract_embeddings(net,train_loader,ms = scales,msp = 1.0,print_freq=20000)
	    print("Train descriptors finished...")
	'''
    

    descriptors_dict = {}
    descriptors_dict["train_descriptors"] = np.array(train_descr).astype("float32")
    '''
    if not args.queries_only:
	    descriptors_dict["train_descriptors"] = np.array(train_descr).astype("float32")
    '''
    descriptors_dict["test_descriptors"] = np.array(test_descr).astype("float32")
    descriptors_dict["val_descriptors"] = np.array(val_descr).astype("float32")

	#save descriptors
    with open(exp_dir+"descriptors.pkl", 'wb') as data:
	    pickle.dump(descriptors_dict,data,protocol = pickle.HIGHEST_PROTOCOL)
	    print("descriptors pickle file complete: {}".format(exp_dir+"descriptors.pkl"))



if __name__ == '__main__':
	main()