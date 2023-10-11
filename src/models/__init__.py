from .archs import *
import torchvision
import torch
import torch.nn
import os
from torch.autograd import Variable


def build_model(params):
    head = None
    n_classes = {'gtsrb': 43, 'cifar10': 10, 'pubfig':65, 'cifar100':100, 'ytfaces':1283, 'ytfaces_plus':1293, 'scrub': 531, 'scrub_plus': 541}
    if params.num_classes > -1:
        n_classes[params.dataset] = params.num_classes
        print(f"adjust classes to {params.num_classes}")
    if params.model == 'simple':
        model = simpleNN(num_classes=n_classes[params.dataset])
    elif params.model == 'resnet18':
        model = ResNet18(num_classes=n_classes[params.dataset])
    elif params.model == 'resnet50':
        model = ResNet50(num_classes=n_classes[params.dataset])
    elif params.model == 'sphereface':
        from .net_sphere import sphere20a
        model = sphere20a()
        if not os.path.exists('./sphere20a_20171020.pth'):
            assert False == True, 'Please download SphereFace checkpoint and place in /src/models/ - see README for details'
        model.load_state_dict(torch.load('./sphere20a_20171020.pth'))
        new_angle = net_sphere.AngleLinear(512, n_classes[params.dataset]) # 65 classes in pubfig    
        model.fc6 = new_angle
        model.fc6.requires_grad = True
        pms = list(model.parameters())
        for el in pms[:-params.num_unfreeze]:
            el.requires_grad = False # everything but last layer is frozen. 
    else:
        print("failed to initialize model")
        assert False == True

    # Make sure the model is on the GPU
    model.cuda()
    
    if 'ytfaces' in params.dataset and params.model == 'resnet50':
        # adjustment to make resnet50 work with ytfaces
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, n_classes[params.dataset]))
        model.fc = model.fc.cuda()


    return model, head

### Additional functions for FR ### 

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index).to(torch.bool)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn
    
