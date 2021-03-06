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
from metric import dc, assd, hd
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


def roll_img(x, n):
    return torch.cat((x[:,:,:,-n:], x[:,:,:, :-n]), dim=-1)
def roll_pred(x, n):
    return torch.cat((x[:,-n:], x[:, :-n]), dim=-1)

def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)

# train


def train(model, criterion, optimizer, input_img_gt, hps):
    model.train()
    output = model(input_img_gt['img'], U_net_only=False)
    criterion_l1 = nn.L1Loss()
    
    if hps.learning.loss == "MSELoss" or hps.learning.loss == "L1Loss":
        loss = criterion(output, input_img_gt['gt'])
        loss_l1 = criterion_l1(output, input_img_gt['gt'])
    else:
        raise AttributeError('Loss not implemented!')
    # watch l1
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_l1.detach().cpu().numpy()
# val


def val(model, criterion, input_img_gt, hps):
    model.eval()
    output = model(input_img_gt['img'], U_net_only=False)
    criterion_l1 = nn.L1Loss()

    if hps.learning.loss == "MSELoss" or hps.learning.loss == "L1Loss":
        loss = criterion(output, input_img_gt['gt'])
        loss_l1 = criterion_l1(output, input_img_gt['gt'])
    else:
        raise AttributeError('Loss not implemented!')
    
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

    # IVUSdiv = IVUSDivide(
    #     hps.learning.data.gt_dir_prefix, hps.learning.data.img_dir_prefix, tr_ratio=hps.learning.data.tr_ratio)
    # case_list = IVUSdiv(surf=hps.learning.data.surf, seed=hps.learning.data.seed)
    # aug_dict = {"saltpepper": SaltPepperNoise(sp_ratio=0.05), 
    #             "Gaussian": AddNoiseGaussian(loc=0, scale=0.1),
    #             "cropresize": RandomCropResize(crop_ratio=0.9), 
    #             "circulateud": CirculateUD(),
    #             "mirrorlr":MirrorLR(), 
    #             "circulatelr": CirculateLR()}
    # rand_aug = RandomApplyTrans(trans_seq=[aug_dict[i] for i in hps.learning.augmentation],
    #                             trans_seq_post=[NormalizeSTD()],
    #                             trans_seq_pre=[NormalizeSTD()])
    # val_aug = RandomApplyTrans(trans_seq=[],
    #                             trans_seq_post=[NormalizeSTD()],
    #                             trans_seq_pre=[NormalizeSTD()])
    # tr_dataset = IVUSDataset(case_list['tr_list'], transform=rand_aug, gaus_gt=False)
    # tr_loader = DataLoader(tr_dataset, shuffle=True,
    #                        batch_size=hps.learning.batch_size, num_workers=0)
    # val_dataset = IVUSDataset(case_list['val_list'], transform=val_aug, gaus_gt=False)
    # val_loader = DataLoader(val_dataset, shuffle=True,
    #                         batch_size=hps.learning.batch_size, num_workers=0)
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

    optimizer_unet = getattr(optim, hps.learning.optimizer)(
        [{'params': model.U_net.parameters(), 'lr': hps.learning.lr_unet},])
    optimizer_smoother  = getattr(optim, hps.learning.optimizer)(
         [{'params': model.w_comp, 'lr': hps.learning.lr_smoother}])
    
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

    epoch = 0
    best_loss = hps.learning.best_loss

    for epoch_tmp in range(0, hps.learning.total_iterations):
        for epoch_unet_tmp in range(0, hps.learning.unet_iterations):
            tr_loss = 0
            tr_mb = 0
            for step, batch in enumerate(tr_loader):
                batch = {key: value.float().cuda()
                        for (key, value) in batch.items() if key not in ["gt_dir", "cartpolar", "img_dir"]}
                m_batch_loss = train(model, loss_func, optimizer_unet, batch, hps)
                tr_loss += m_batch_loss
                tr_mb += 1
            epoch_tr_loss = tr_loss / tr_mb
            writer.add_scalar('data/train_loss', epoch_tr_loss, epoch)
            print("Epoch: " + str(epoch))
            print("     tr_loss: " + "%.5e" % epoch_tr_loss)
            epoch += 1

        for epoch_smoother_tmp in range(0, hps.learning.smoother_iterations):
            tr_loss = 0
            tr_mb = 0
            for step, batch in enumerate(val_loader):
                batch = {key: value.float().cuda()
                        for (key, value) in batch.items() if key not in ["gt_dir", "cartpolar", "img_dir"]}
                m_batch_loss = train(model, loss_func, optimizer_smoother, batch, hps)
                tr_loss += m_batch_loss
                tr_mb += 1
            epoch_tr_loss = tr_loss / tr_mb
            writer.add_scalar('data/train_loss', epoch_tr_loss, epoch)
            w_comp = model.w_comp.detach().cpu().numpy()
            writer.add_scalar('data/w_comp', w_comp)
            print("Epoch: " + str(epoch))
            print("     tr_loss: " + "%.5e" % epoch_tr_loss + " w_comp: " + "%.5e" % w_comp)
            epoch += 1

        val_loss = 0
        val_mb = 0
        for step, batch in enumerate(val_loader):
            batch = {key: value.float().cuda()
                     for (key, value) in batch.items() if key not in ["gt_dir", "cartpolar", "img_dir"]}
            m_batch_loss = val(model, loss_func, batch, hps)
            val_loss += m_batch_loss
            val_mb += 1
        epoch_val_loss = val_loss / val_mb
        writer.add_scalar('data/val_loss', epoch_val_loss, epoch_tmp)
        print("     val_loss: " + "%.5e" % epoch_val_loss)

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss
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
    combo_dir = "../../instance_seg/data/MoNuSeg/train_data/combo/"
    test_dataset = MoNuDataset(polar_img_np=os.path.join(combo_dir, "test_polar_img.npy"), 
                            polar_gt_np=os.path.join(combo_dir, "test_polar_gt.npy"),
                            img_np = os.path.join(combo_dir, "test_img.npy"),
                            gt_np = os.path.join(combo_dir, "test_gt.npy")
                            )
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
    print(model.w_comp.detach().cpu().numpy())
        # batch_img = batch['img'].float().cuda()
        # pred = model(batch_img, U_net_only=hps.network.unet_only)
        # pred = pred.squeeze().detach().cpu().numpy()
    rept_nb = 8
    dc_list, assd_list, hd_list = [], [], []
    for step, batch in enumerate(test_loader):
        pred = np.zeros(64, dtype=np.float32)
        for shift_nb in range(rept_nb):
            batch_img = batch['img'].float().cuda()
            if shift_nb==0:
                pred_tmp = model(batch_img, U_net_only=hps.network.unet_only)
                pred_tmp = pred_tmp.squeeze().detach().cpu().numpy()
                pred += pred_tmp
            else:
                shift = int(shift_nb*64/rept_nb)
                batch_img = roll_img(batch_img, shift)
                pred_tmp = model(batch_img, U_net_only=hps.network.unet_only)
                pred_tmp = roll_pred(pred_tmp, -shift)
                pred_tmp = pred_tmp.squeeze().detach().cpu().numpy()
                pred += pred_tmp
        pred = 1.*pred/rept_nb
        # convert to cart
        img = batch['cart_img'][0].detach().numpy()
        # print(img.shape)
        gt = batch['cart_gt'][0].detach().numpy()
        phy_radius = 0.5*np.sqrt(np.average(np.array(img.shape[:-1])**2)) - 1
        cartpolar = CartPolar(np.array(img.shape[:-1])/2.,
                            phy_radius, 64, 32)
        pred = cartpolar.gt2cart(pred)
        # compute metrics
        # region filling pred
        pred_array = [np.expand_dims(np.swapaxes(np.rint(np.array(pred)).astype(int), 0, 1), axis=1)]
        # print("contours shape: ", pred_array[0].shape)
        # print("gt shape: ", gt.shape)
        pred_img = np.zeros((146, 146, 3), dtype=np.uint8)
        cv2.drawContours(pred_img, pred_array, -1, (255,255,255), cv2.FILLED)
        pred_img = pred_img[:,:,0]
        DICE = dc(gt, pred_img)
        ASSD = assd(gt, pred_img)
        HD = hd(gt, pred_img)
        dc_list.append(DICE)
        assd_list.append(ASSD)
        hd_list.append(HD)
        print("dc: ", DICE)
        print("assd: ", ASSD)
        print("hd: ", HD)

        # gt = cartpolar.gt2cart(gt)
        if not os.path.isdir(hps.test.pred_dir):
            os.mkdir(hps.test.pred_dir)
        pred_dir = os.path.join(hps.test.pred_dir,"%d.png" % step)
        f, ax = plt.subplots(1,2)
        ax[0].imshow(img.astype(np.uint8))
        ax[1].imshow(img.astype(np.uint8))
        ax[1].plot(pred[0], pred[1], 'r-')
        ax[1].contour(gt, levels=[0,1], colors='green')
        f.savefig(pred_dir, bbox_inches='tight')
        plt.close()
    stat_dir = os.path.join(hps.test.pred_dir,"results.txt")
    dc_list = np.array(dc_list)
    assd_list = np.array(assd_list)
    hd_list = np.array(hd_list)
    output = np.stack([dc_list.mean(), dc_list.std(), assd_list.mean(), assd_list.std(), hd_list.mean(), hd_list.std()])
    np.savetxt(stat_dir, output, delimiter=',')
    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    time_slice = 1.*time_elapsed/step
    print("Time for each slice: %e" % time_slice)



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
