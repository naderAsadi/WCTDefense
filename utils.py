from __future__ import division
import torch
from sklearn.neighbors import KNeighborsClassifier
import pickle

class NearestNeighbor(object):
    """
    Nearest-Neighbor algorithm using SkLearn
    :return nearest sample to test images
    """
    def __init__(self):
        super(NearestNeighbor, self).__init__()
        self.nn = KNeighborsClassifier(n_neighbors=1)

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.nn.fit(x_train, y_train)

    def load(self, root):
        with open(root, 'rb') as handle:
            data = pickle.load(handle)
        self.train(data[0], data[1])
        return

    def predict(self, x):
        _, ind = self.nn.kneighbors(x)
        return self.x_train[ind]

##################################
# Functions for WCT
##################################
def whiten_and_color(cF, sF):
    cFSize = cF.size()
    c_mean = torch.mean(cF, 1)  # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean  # centering fc by subtracting its mean vector

    contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double()
    c_u, c_e, c_v = torch.svd(contentConv, some=False)  # c_v = orthogonal matrix of eigenvectors

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF, 1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
    s_u, s_e, s_v = torch.svd(styleConv, some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cF)

    s_d = (s_e[0:k_s]).pow(0.5)

    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature, whiten_cF  # , color_fm


def transform(cF, sF, csF, alpha):
    cF = cF.double()
    sF = sF.double()
    C, W, H = cF.size(0), cF.size(1), cF.size(2)
    _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
    cFView = cF.view(C, -1)
    sFView = sF.view(C, -1)

    targetFeature, whiten = whiten_and_color(cFView, sFView)
    targetFeature = targetFeature.view_as(cF)
    ccsF = alpha * targetFeature + (1.0 - alpha) * cF
    ccsF = ccsF.float().unsqueeze(0)
    csF.data.resize_(ccsF.size()).copy_(ccsF)
    return csF, whiten  # , whittened
