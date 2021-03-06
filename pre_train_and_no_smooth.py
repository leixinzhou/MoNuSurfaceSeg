from torch.utils.data import DataLoader
from MoNuDataset import *
from nets import SurfNet
from hpara_io import json2obj
import argparse
import time
from tensorboardX import SummaryWriter
from torch import optim
from torch import nn
import shutil
import os
from cartpolar import CartPolar
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)

# train


def train(model, criterion, optimizer, input_img_gt, hps):
    model.train()
    output = model(input_img_gt['img'], U_net_only=hps.network.unet_only)
    criterion_l1 = nn.L1Loss()
    if hps.learning.loss == "KLDivLoss":
        loss = criterion(output, input_img_gt['gaus_gt'].permute(0, 2, 1))
        # print(input_img_gt['img'].size())
    elif hps.learning.loss == "MSELoss" or hps.learning.loss == "L1Loss":
        loss = criterion(output, input_img_gt['gt'])
        loss_l1 = criterion_l1(output, input_img_gt['gt'])
    else:
        raise AttributeError('Loss not implemented!')
    # watch l1
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if hps.learning.loss == "KLDivLoss":
        return loss.detach().cpu().numpy()
    else:
        return loss_l1.detach().cpu().numpy()
# val


def val(model, criterion, input_img_gt, hps):
    model.eval()
    output = model(input_img_gt['img'], U_net_only=hps.network.unet_only)
    criterion_l1 = nn.L1Loss()

    if hps.learning.loss == "KLDivLoss":
        loss = criterion(output, input_img_gt['gaus_gt'].permute(0, 2, 1))
    elif hps.learning.loss == "MSELoss" or hps.learning.loss == "L1Loss":
        loss = criterion(output, input_img_gt['gt'])
        loss_l1 = criterion_l1(output, input_img_gt['gt'])
    else:
        raise AttributeError('Loss not implemented!')

    if hps.learning.loss == "KLDivLoss":
        return loss.detach().cpu().numpy()
    else:
        return loss_l1.detach().cpu().numpy()
# learn


def learn(model, hps):
    since = time.time()
    writer = SummaryWriter(hps.learning.checkpoint_path)
    if hps.gpu_nb >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hps.gpu_nb)
        model.cuda()
    else:
        raise NotImplementedError("CPU version is not implemented!")

   
    aug_dict = {"saltpepper": SaltPepperNoise(sp_ratio=0.05), 
                "Gaussian": AddNoiseGaussian(loc=0, scale=0.1),
                "cropresize": RandomCropResize(crop_ratio=0.9), 
                "circulateud": CirculateUD(),
                "mirrorlr":MirrorLR(), 
                "circulatelr": CirculateLR()}
    # rand_aug = RandomApplyTrans(trans_seq=[aug_dict[i] for i in hps.learning.augmentation],
    #                             trans_seq_post=[NormalizeSTD()],
    #                             trans_seq_pre=[NormalizeSTD()])
    # rand_aug = RandomApplyTrans(trans_seq=[],
    #                             trans_seq_post=[NormalizeSTD()],
    #                             trans_seq_pre=[NormalizeSTD()])
    # val_aug = RandomApplyTrans(trans_seq=[],
    #                             trans_seq_post=[NormalizeSTD()],
    #                             trans_seq_pre=[NormalizeSTD()])
    combo_dir = "../../instance_seg/data/MoNuSeg/train_data/combo/"
    tr_dataset = MoNuDataset(polar_img_np=os.path.join(combo_dir, "tr_polar_img.npy"), 
                            polar_gt_np=os.path.join(combo_dir, "tr_polar_gt.npy")
                            )
    tr_loader = DataLoader(tr_dataset, shuffle=True,
                           batch_size=hps.learning.batch_size, num_workers=0)
    val_dataset = MoNuDataset(polar_img_np=os.path.join(combo_dir, "val_polar_img.npy"), 
                            polar_gt_np = os.path.join(combo_dir, "val_polar_gt.npy")
                            )
    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=hps.learning.batch_size, num_workers=0)

    optimizer = getattr(optim, hps.learning.optimizer)(
        [{'params': model.U_net.parameters(), 'lr': hps.learning.lr_unet},
         {'params': model.w_comp, 'lr': hps.learning.lr_smoother}])
    scheduler = getattr(optim.lr_scheduler,
                        hps.learning.scheduler)(optimizer, factor=hps.learning.scheduler_params.factor,
                                                patience=hps.learning.scheduler_params.patience,
                                                threshold=hps.learning.scheduler_params.threshold,
                                                threshold_mode=hps.learning.scheduler_params.threshold_mode)
    try:
        loss_func = getattr(nn, hps.learning.loss)()
    except AttributeError:
        raise AttributeError(hps.learning.loss+" is not implemented!")
    # criterion_KLD = torch.nn.KLDivLoss()

    if os.path.isfile(hps.learning.resume_path):
        print('loading checkpoint: {}'.format(hps.learning.resume_path))
        checkpoint = torch.load(hps.learning.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(hps.learning.resume_path))

    epoch_start = 0
    best_loss = hps.learning.best_loss

    for epoch in range(epoch_start, hps.learning.total_iterations):
        writer.add_scalar(
            'data/train_lr', optimizer.param_groups[0]['lr'], epoch)
        tr_loss = 0
        tr_mb = 0
        for step, batch in enumerate(tr_loader):
            batch = {key: value.float().cuda() for (key, value) in batch.items() }
            m_batch_loss = train(model, loss_func, optimizer, batch, hps)
            tr_loss += m_batch_loss
            tr_mb += 1
        epoch_tr_loss = tr_loss / tr_mb
        # assert epoch_tr_loss == 0
        writer.add_scalar('data/train_loss', epoch_tr_loss, epoch)
        print("Epoch: " + str(epoch))
        print("     tr_loss: " + "%.5e" % epoch_tr_loss)
        scheduler.step(epoch_tr_loss)

        val_loss = 0
        val_mb = 0
        for step, batch in enumerate(val_loader):
            batch = {key: value.float().cuda()
                     for (key, value) in batch.items() }
            m_batch_loss = val(model, loss_func, batch, hps)
            val_loss += m_batch_loss
            val_mb += 1
        epoch_val_loss = val_loss / val_mb
        writer.add_scalar('data/val_loss', epoch_val_loss, epoch)
        print("     val_loss: " + "%.5e" % epoch_val_loss)

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict()
                },
                path=hps.learning.checkpoint_path,
            )

    writer.export_scalars_to_json(os.path.join(
        hps.learning.checkpoint_path, "all_scalars.json"))
    writer.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

def infer(model, hps):
    since = time.time()
    if hps.gpu_nb >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hps.gpu_nb)
        model.cuda()
    else:
        raise NotImplementedError("CPU version is not implemented!")
    rand_aug = RandomApplyTrans(trans_seq=[],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])
    test_dataset = MoNuDataset(img_np=os.path.join(combo_dir, "test_img.npy"), 
                            gt_np = os.path.join(combo_dir, "test_gt.npy"),
                             transform=rand_aug, gaus_gt=False)
    test_loader = DataLoader(test_dataset, shuffle=False,
                           batch_size=hps.test.batch_size, num_workers=0)
    
    if os.path.isfile(hps.test.resume_path):
        print('loading checkpoint: {}'.format(hps.test.resume_path))
        checkpoint = torch.load(hps.test.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(hps.test.resume_path))
    model.eval()
    rept_nb = 8
    ROW_LEN = 64
    COL_LEN = 64
    for step, batch in enumerate(test_loader):
        pred = np.zeros(ROW_LEN, dtype=np.float32)
        for shift_nb in range(rept_nb):
            batch_img = batch['img'].float().cuda()
            if shift_nb==0:
                pred_tmp = model(batch_img, U_net_only=hps.network.unet_only)
                pred_tmp = pred_tmp.squeeze().detach().cpu().numpy()
                pred += pred_tmp
            else:
                shift = int(shift_nb*ROW_LEN/rept_nb)
                batch_img = roll_img(batch_img, shift)
                pred_tmp = model(batch_img, U_net_only=hps.network.unet_only)
                pred_tmp = roll_pred(pred_tmp, -shift)
                pred_tmp = pred_tmp.squeeze().detach().cpu().numpy()
                pred += pred_tmp
        pred = 1.*pred/rept_nb
        # convert to cart
        img = plt.imread(batch['img_dir'][0])
        phy_radius = 0.5*np.sqrt(np.average(np.array(img.shape)**2)) - 1
        cartpolar = CartPolar(np.array(img.shape)/2.,
                              phy_radius, 256, 128)
        pred = cartpolar.gt2cart(pred)
        if not os.path.isdir(hps.test.pred_dir):
            os.mkdir(hps.test.pred_dir)
        pred_dir = os.path.join(hps.test.pred_dir,batch['gt_dir'][0].split("/")[-1])
        pred = np.transpose(np.stack(pred, axis=0))
       
        np.savetxt(pred_dir, pred, delimiter=',')
    print("Test done!")


def main():
    # read configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyperparams', default='./para/hparas_unet.json',
                        type=str, metavar='FILE.PATH',
                        help='path to hyperparameters setting file (default: ./para/hparas_unet.json)')

    args = parser.parse_args()
    try:
        hp_data = open(args.hyperparams).read()
    except IOError:
        print('Couldn\'t read hyperparameter setting file')
    hps = json2obj(hp_data)

    net = SurfNet(depth=hps.network.depth, start_filts=hps.network.start_filts, )
    if hps.inference_mode:
        infer(net, hps)
    else:
        try:
            learn(net, hps)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), os.path.join(
                hps.learning.checkpoint_path, 'INTERRUPTED.pth'))
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


if __name__ == '__main__':
    main()
