import os
import torch
import torch.utils.data
from torch.autograd import Variable
import torchvision

from models.models import AEVGG19, VGG16, load_model
from utils import transform, NearestNeighbor
from loader import get_loader, Dataset
from advertorch.attacks import FGSM, PGDAttack, CarliniWagnerL2Attack, JSMA, LinfBasicIterativeAttack


def styleTransfer(contentImg,styleImg,imname,csF):
    sF4 = model.encode(styleImg)  # style
    cF4 = model.encode(contentImg)  # content
    sF4_ = sF4.data.cpu().squeeze(0)
    cF4_ = cF4.data.cpu().squeeze(0)
    csF4, whiten = transform(cF4_, sF4_, csF, alpha=1)
    #print(whiten.shape, csF4.shape)

    out = model.decode(whiten.type(torch.float).view(csF4.shape).to(device))
    #out = model.decode(cF4.to(device))

    # save_image has this wired design to pad images with 4 pixels at default.
    #print(csF4.shape)
    torchvision.utils.save_image(out.data,os.path.join('./results/mnist/fgsm/whiten/',imname))
    return

def compute_mean_cov(cF):
    cF = cF.double()
    C, W, H = cF.size(0), cF.size(1), cF.size(2)
    cF = cF.view(C, -1)

    cFSize = cF.size()
    c_mean = torch.mean(cF, 1)  # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean  # centering fc by subtracting its mean vector

    contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double()

    return c_mean, contentConv


# def test_model(model, dataroot, device):
#     ################
#     # Data Loader
#     ################
#     test_loader = get_loader('mnist', 32, batch_size=1, dataroot=dataroot, train=False)
#
#     print('-----> Testing <-----')
#     correct_nn, correct_net = 0, 0
#
#     m = torch.distributions.MultivariateNormal(torch.zeros(2, 2), torch.eye(2, 2))
#     s = m.sample().unsqueeze(0)
#     for i in range(511):
#         s = torch.cat((s, m.sample().unsqueeze(0)))
#     print(s.shape)
#
#     total = 0
#     csF = torch.Tensor()
#     csF = Variable(csF)
#     for i, (image, label) in enumerate(test_loader, 1):
#         image, label = image.to(device), label.to(device)
#         # sF4 = style.view(img.shape)
#         # sF4 = model.forward4(sF4)
#         # sF4 = model.forward5(sF4)
#         sF4 = model.encode3(image)  # style
#
#         mean, cov = compute_mean_cov(sF4)
#         print(mean.shape, cov.shape)
#
#         #cF4 = s.unsqueeze(0).to(device)  # content
#         #sF4_ = sF4.data.cpu().squeeze(0)
#         #cF4_ = cF4.data.cpu().squeeze(0)
#         #csF4 = transform(cF4_, sF4_, csF, alpha=1).to(device)
#         #out = model.decode3(csF4)
#
#         #out = model(image)
#
#         #_, predicted = torch.max(out.data, 1)
#         #correct_net += (predicted == label ).sum().item()
# #
#         #total += label.size(0)
#
#         # if predicted == label and predicted.item() != np.argmax(nn_out):
#         #    data.append([pert,style, predicted, np.argmax(nn_out)])
#
#         #if i % 100 == 0:
#         #    print('Iter: {} - Acc NN: {} - Acc Net: {}'.format(i, correct_nn / total, correct_net / total))
#
#     #accuracy = correct / total
#     #print('=======> Test Accuracy: {}'.format(correct_net / total))
#     #return accuracy

def test_model(model, dataroot, device):
    ################
    # Data Loader
    ################
    #test_loader = get_loader('mnist', 32, batch_size=1, dataroot=dataroot, train=False)
    dataset = Dataset('./results/mnist/fgsm/adv', './results/mnist/fgsm/clean', 256)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    #fgsm = FGSM(predict=model, eps=0.3)
    #pgd = PGDAttack(predict=model)
    #cw = CarliniWagnerL2Attack(predict=model, num_classes=10)
    #jsma = JSMA(predict=model, num_classes=10)
    ###################
    #Nearest Neighbor
    ###################
    #nn = NearestNeighbor()
    #nn.load('./cifar_clean_encode0.pickle')
    csF = torch.Tensor()
    csF = Variable(csF)
    for i, (contentImg, styleImg, imname) in enumerate(loader):
        imname = imname[0]
        print('Transferring ' + imname)
        contentImg = contentImg.cuda()
        styleImg = styleImg.cuda()
        cImg = Variable(contentImg)
        sImg = Variable(styleImg)
        # WCT Style Transfer
        styleTransfer(cImg, sImg, imname, csF)


def attack(model, dataroot, device):
    ################
    # Data Loader
    ################
    test_loader = get_loader('letters', 32, batch_size=1, dataroot=dataroot, train=False)

    #fgsm = FGSM(predict=model, eps=0.3)
    #bim = LinfBasicIterativeAttack(predict=model)
    #pgd = PGDAttack(predict=model, eps=0.8)

    total, correct = 0, 0
    for i, (image, label) in enumerate(test_loader, 1):
        image, label = image.to(device), label.to(device)
        #pert = fgsm.perturb(image)

        #out = model(pert)

        torchvision.utils.save_image(image.data, './results/letters/{}.png'.format(i))
        #torchvision.utils.save_image(pert.data, './results/mnist/fgsm/adv/{}.png'.format(i))

        #_, predicted = torch.max(out.data, 1)
        #correct += (predicted == label - 1).sum().item()
        #total += label.size(0)
        #if i == 100:
        #    break

    print('=======> Test Accuracy: {}'.format(correct / total))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AEVGG19().to(device)
    #model = load_model(model, './saved_models/letters_vgg16.pth')
    #model = load_model(model, '/home/nader/.torch/models/VGG19_AE.pth')
    attack(model, dataroot='~/AI/Datasets/letters', device=device)


# loader = torch.utils.data.DataLoader(
#            torchvision.datasets.ImageFolder(root='./results/mnist/fgsm/recons',
#                                       transform=torchvision.transforms.Compose([
#                                           torchvision.transforms.Resize(32),
#                                           torchvision.transforms.ToTensor(),
#                                       ])),
#            batch_size=1, shuffle=False)
#
# for i, image in enumerate(loader, 0):
#   image = image[0]
#   image = 0.2989*image[:,0,:,:] + 0.5870*image[:,1,:,:] + 0.1140*image[:,2,:,:]
#   torchvision.utils.save_image(image.data, './results/mnist/fgsm/recons/{}.png'.format(i))
