import argparse
import torch
import itertools
from model import Generator
from model import Discriminator
from utils import weights_init_normal
from utils import LambdaLR
from torch.autograd import Variable
from utils import ReplayBuffer
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from dataset import ImageDataset
from utils import Logger
import os


def loss_function():
    GAN_loss = torch.nn.MSELoss()
    Cycle_loss = torch.nn.L1Loss()
    Identity_loss = torch.nn.L1Loss()
    return GAN_loss, Cycle_loss, Identity_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='training epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--data_root', type=str, default='dataset/horse2zebra/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cup threads to use during batch generation')
    parser.add_argument("--pre_train", action='store_true', help='continue to train model')
    parser.add_argument('--model_info', type=str, default='output/model_info.pth', help='Model information file')
    parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth',
                        help='B2A generator checkpoint file')
    parser.add_argument('--discriminator_A', type=str, default='output/netD_A.pth',
                        help='Discriminator_A checkpoint file')
    parser.add_argument('--discriminator_B', type=str, default='output/netD_B.pth',
                        help='Discriminator_B checkpoint file')
    opt = parser.parse_args()

    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr,
                                   betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)

    # try to load the pre_train to continue the last time training
    if opt.pre_train and os.path.exists(opt.model_info):
        # load the pre model
        netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
        netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
        netD_A.load_state_dict(torch.load(opt.discriminator_A))
        netD_B.load_state_dict(torch.load(opt.discriminator_B))

        # load the pre optimizer
        model_info = torch.load(opt.model_info)
        optimizer_G.load_state_dict(model_info['optimizer_G'])
        optimizer_D_A.load_state_dict(model_info['optimizer_D_A'])
        optimizer_D_B.load_state_dict(model_info['optimizer_D_B'])

        # load the pre lr_scheduler
        lr_scheduler_G.load_state_dict(model_info['lr_scheduler_G'])
        lr_scheduler_D_A.load_state_dict(model_info['lr_scheduler_D_A'])
        lr_scheduler_D_B.load_state_dict(model_info['lr_scheduler_D_B'])

        # load the epoch
        opt.epoch = model_info['epoch']



    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    criterion_GAN, criterion_cycle, criterion_identity = loss_function()

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_ = [
        transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    dataloader = data.DataLoader(ImageDataset(opt.data_root, transforms_=transforms_, unaligned=True),
                                 batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    logger = Logger(opt.epoch, opt.n_epochs, len(dataloader))

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # ----------Generators A2B and B2A----------
            optimizer_G.zero_grad()

            # Identity loss:
            #   it is helpful to introduce an additional loss to encourage the mapping to preserve color composition between the input and output.
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            loss_cycle = loss_cycle_ABA + loss_cycle_BAB

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_cycle + loss_GAN_A2B + loss_GAN_B2A
            loss_G.backward()
            optimizer_G.step()

            # ----------Discriminator A----------
            optimizer_D_A.zero_grad()
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # ----------Discriminator B----------
            optimizer_D_B.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # Progress report (http://localhost:8097)
            logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                        'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # save info of the train
        torch.save(
            {"epoch": epoch+1, "optimizer_G": optimizer_G.state_dict(), "optimizer_D_A": optimizer_D_A.state_dict(), "optimizer_D_B": optimizer_D_B.state_dict(),
             "lr_scheduler_G": lr_scheduler_G.state_dict(), "lr_scheduler_D_A": lr_scheduler_D_A.state_dict(),
             "lr_scheduler_D_B": lr_scheduler_D_B.state_dict()}, opt.model_info)
        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), opt.generator_A2B)
        torch.save(netG_B2A.state_dict(), opt.generator_B2A)
        torch.save(netD_A.state_dict(), opt.discriminator_A)
        torch.save(netD_B.state_dict(), opt.discriminator_B)
