import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from Architecture.DATransUNet import DA_Transformer as ViT_seg
from Architecture.DATransUNet import CONFIGS as CONFIGS_ViT_seg
import sys
import time
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--val_interval', type=int, default=0,
                    help='validate every N epochs and save best_model.pth (0 = disabled)')
args = parser.parse_args()


def _validate_synapse(args, model):
    from datasets.dataset_synapse import Synapse_dataset
    from utils import test_single_volume
    vol_path = os.path.join(os.path.dirname(args.root_path), 'test_vol_h5')
    db_val = Synapse_dataset(base_dir=vol_path, split="test_vol", list_dir=args.list_dir)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    net = model.module if hasattr(model, 'module') else model
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for sample in val_loader:
            metric_i = test_single_volume(
                sample["image"], sample["label"], net,
                classes=args.num_classes, patch_size=[args.img_size, args.img_size])
            metric_list += np.array(metric_i)
    model.train()
    return float(np.mean(metric_list / len(db_val), axis=0)[0])


def _validate_2d(args, model, dataset_class):
    from utils import test_single_volume
    db_val = dataset_class(base_dir=args.root_path, split="test", list_dir=args.list_dir)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    net = model.module if hasattr(model, 'module') else model
    net.eval()
    metric_list = 0.0
    with torch.no_grad():
        for sample in val_loader:
            metric_i = test_single_volume(
                sample["image"], sample["label"], net,
                classes=args.num_classes, patch_size=[args.img_size, args.img_size])
            metric_list += np.array(metric_i)
    net.train()
    return float(np.mean(metric_list / len(db_val), axis=0)[0])


def _trainer_2d(args, model, snapshot_path, db_train, dataset_class):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0,
                             pin_memory=True, worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    train_start = time.time()
    val_time_s = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch_num in iterator:
        epoch_loss_sum = 0.0
        epoch_loss_ce_sum = 0.0
        epoch_batches = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            epoch_loss_sum += loss.item()
            epoch_loss_ce_sum += loss_ce.item()
            epoch_batches += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

        elapsed_h = (time.time() - train_start - val_time_s) / 3600
        avg_loss = epoch_loss_sum / epoch_batches
        avg_loss_ce = epoch_loss_ce_sum / epoch_batches
        logging.info("epoch %d/%d  loss: %.4f  loss_ce: %.4f  lr: %.6f  elapsed: %.2fh" % (
            epoch_num + 1, max_epoch, avg_loss, avg_loss_ce, lr_, elapsed_h))

        save_interval = 50
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if args.val_interval > 0 and (epoch_num + 1) % args.val_interval == 0:
            _val_t0 = time.time()
            val_dice = _validate_2d(args, model, dataset_class)
            val_time_s += time.time() - _val_t0
            logging.info("Validation DSC: %.4f  best: %.4f" % (val_dice, best_performance))
            if val_dice > best_performance:
                best_performance = val_dice
                net = model.module if hasattr(model, 'module') else model
                torch.save(net.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
                logging.info("=> best model saved (DSC %.4f)" % best_performance)

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    total_hours = (time.time() - train_start - val_time_s) / 3600
    logging.info("Total training time: {:.2f} hours (excl. {:.1f} min validation overhead)".format(
        total_hours, val_time_s / 60))
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        logging.info("Peak VRAM: {:.0f} MB ({:.1f} GB)".format(peak_mb, peak_mb / 1024))
    writer.close()
    return "Training Finished!"


def trainer_kvasir(args, model, snapshot_path):
    from datasets.dataset_kvasir import Kvasir_dataset, RandomGenerator
    db_train = Kvasir_dataset(
        base_dir=args.root_path, list_dir=args.list_dir, split="train",
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]))
    return _trainer_2d(args, model, snapshot_path, db_train, Kvasir_dataset)


def trainer_isic(args, model, snapshot_path):
    from datasets.dataset_isic import ISIC_dataset, RandomGenerator
    db_train = ISIC_dataset(
        base_dir=args.root_path, list_dir=args.list_dir, split="train",
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]))
    return _trainer_2d(args, model, snapshot_path, db_train, ISIC_dataset)


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0,
                             pin_memory=True, worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    train_start = time.time()
    val_time_s = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch_num in iterator:
        epoch_loss_sum = 0.0
        epoch_loss_ce_sum = 0.0
        epoch_batches = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            epoch_loss_sum += loss.item()
            epoch_loss_ce_sum += loss_ce.item()
            epoch_batches += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        elapsed_h = (time.time() - train_start - val_time_s) / 3600
        avg_loss = epoch_loss_sum / epoch_batches
        avg_loss_ce = epoch_loss_ce_sum / epoch_batches
        logging.info("epoch %d/%d  loss: %.4f  loss_ce: %.4f  lr: %.6f  elapsed: %.2fh" % (
            epoch_num + 1, max_epoch, avg_loss, avg_loss_ce, lr_, elapsed_h))

        save_interval = 50  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if args.val_interval > 0 and (
                (epoch_num + 1) % args.val_interval == 0 or epoch_num >= max_epoch - 1):
            _val_t0 = time.time()
            val_dice = _validate_synapse(args, model)
            val_time_s += time.time() - _val_t0
            logging.info("epoch %d val_dice: %.4f  best: %.4f" % (
                epoch_num + 1, val_dice, best_performance))
            if val_dice > best_performance:
                best_performance = val_dice
                net = model.module if hasattr(model, 'module') else model
                torch.save(net.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
                logging.info("=> best model saved (DSC %.4f)" % best_performance)

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    total_hours = (time.time() - train_start - val_time_s) / 3600
    logging.info("Total training time: {:.2f} hours (excl. {:.1f} min validation overhead)".format(
        total_hours, val_time_s / 60))
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        logging.info("Peak VRAM: {:.0f} MB ({:.1f} GB)".format(peak_mb, peak_mb / 1024))
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Kvasir': {
            'root_path': '../../data/Kvasir-SEG',
            'list_dir': './lists/lists_Kvasir',
            'num_classes': 2,
        },
        'ISIC': {
            'root_path': '../../data/ISIC2018',
            'list_dir': './lists/lists_ISIC',
            'num_classes': 2,
        },
    }
#     if args.batch_size != 24 and args.batch_size % 6 == 0:
#         args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {
        'Synapse': trainer_synapse,
        'Kvasir':  trainer_kvasir,
        'ISIC':    trainer_isic,
    }
    trainer[dataset_name](args, net, snapshot_path)