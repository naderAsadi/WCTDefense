import torch
import torch.utils.data
from torch.autograd import Variable

from models.models import VGG16, load_model
from utils import transform, NearestNeighbor
from loader import get_loader
from advertorch.attacks import FGSM, LinfBasicIterativeAttack, PGDAttack, CarliniWagnerL2Attack

def test_model(model, dataroot, device):
    ################
    # Data Loader
    ################
    test_loader = get_loader('mnist', 32, batch_size=1, dataroot=dataroot)
    ################
    # Attacks
    ################
    fgsm = FGSM(predict=model)
    bim = LinfBasicIterativeAttack(predict=model)
    pgd = PGDAttack(predict=model)
    cw = CarliniWagnerL2Attack(predict=model, num_classes=10)
    ###################
    # Nearest Neighbor
    ###################
    nn = NearestNeighbor()
    nn.load('./mnist_clean.pickle')

    print('-----> Testing <-----')
    correct = 0
    total = 0
    csF = torch.Tensor()
    csF = Variable(csF)
    for i, (image, label) in enumerate(test_loader, 1):
        image, label = image.to(device), label.to(device)
        pert = fgsm.perturb(image)
        #pert = pgd.perturb(image)
        #pert = cw.perturb(image)

        style = nn.predict(image.detach().cpu().view(1,-1))
        style = torch.tensor(style).unsqueeze(0).unsqueeze(0).to(device)

        sF4 = model.encode1(style.view(image.shape))  # style
        cF4 = model.encode1(pert)  # content
        sF4_ = sF4.data.cpu().squeeze(0)
        cF4 = cF4.data.cpu().squeeze(0)
        sF4_ = Variable(sF4_, requires_grad=True)
        csF4 = transform(cF4, sF4_, csF, alpha=1).to(device)

        sF4 = model.forward4(sF4)
        cF4 = model.forward4(csF4)
        sF4_ = sF4.data.cpu().squeeze(0)
        cF4 = cF4.data.cpu().squeeze(0)
        sF4_ = Variable(sF4_, requires_grad=True)
        csF4 = transform(cF4, sF4_, csF, alpha=1).to(device)

        sF4 = model.forward5(sF4)
        cF4 = model.forward5(csF4)
        sF4_ = sF4.data.cpu().squeeze(0)
        cF4 = cF4.data.cpu().squeeze(0)
        sF4_ = Variable(sF4_, requires_grad=True)
        csF4 = transform(cF4, sF4_, csF, alpha=1).to(device)

        out = model.classify(csF4)

        #print(pert.shape)
        #out = model(pert)

        _, predicted = torch.max(out.data, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

        if i % 100 == 0:
            print('Iter: {} - Acc: {}'.format(i, correct / total))

    accuracy = correct / total
    print('=======> Test Accuracy: {}'.format(correct / total))
    return accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16(in_channels=1).to(device)
    model = load_model(model, './saved_models/vgg16_mnist32_refpad.pth')
    test_model(model, dataroot='~/AI/Datasets/mnist/data', device=device)