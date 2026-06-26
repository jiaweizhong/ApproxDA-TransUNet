import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_acdc import ACDC_dataset
from datasets.dataset_kvasir import Kvasir_dataset
from datasets.dataset_isic import ISIC_dataset
from utils import test_single_volume
from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg
try:
    from fvcore.nn import FlopCountAnalysis
    _FLOP_LIB = 'fvcore'
except ImportError:
    try:
        from thop import profile as thop_profile
        _FLOP_LIB = 'thop'
    except ImportError:
        _FLOP_LIB = None

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--window_size', type=int, default=7, help='window size — must match training')
parser.add_argument('--rank', type=int, default=32, help='low-rank dim — must match training')
parser.add_argument('--groups', type=int, default=8, help='channel groups — must match training')
parser.add_argument('--gate_mode', type=str, default='learn',
                    choices=['learn', 'fixed', 'pam', 'cam'],
                    help='Must match training gate_mode')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info("Model parameters: {:.2f} M".format(n_params))

    if _FLOP_LIB:
        try:
            dummy = torch.randn(1, 1, args.img_size, args.img_size).cuda()
            if _FLOP_LIB == 'fvcore':
                _fa = FlopCountAnalysis(model, dummy)
                _fa.unsupported_ops_warnings(False)
                _fa.uncalled_modules_warnings(False)
                logging.info("GFLOPs (single 224x224 slice, fvcore, includes attention bmm): {:.1f}".format(_fa.total() / 1e9))
            else:
                macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
                logging.info("GFLOPs (single 224x224 slice, thop, attention bmm NOT counted): {:.1f}".format(macs / 1e9))
        except Exception as e:
            logging.info("GFLOPs: could not profile ({})".format(e))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.eval()
    metric_list = 0.0
    infer_start = time.time()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    infer_total_s = time.time() - infer_start
    avg_case_ms = infer_total_s / len(db_test) * 1000
    logging.info("Inference total: {:.1f}s  avg per case: {:.1f}ms".format(infer_total_s, avg_case_ms))
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        logging.info("Inference peak VRAM: {:.0f} MB ({:.1f} GB)".format(peak_mb, peak_mb / 1024))

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_iou %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_iou : %f' % (performance, mean_hd95, mean_iou))
    return "Testing Finished!"


if __name__ == "__main__":
    
    if args.is_savenii:
        print('Saving results as NIfTI files.')
    else:
        print('Not saving results.')

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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'ACDC': {
            'Dataset': ACDC_dataset,
            'volume_path': '../data/ACDC/ACDC_training_volumes',
            'list_dir': './lists/lists_ACDC',
            'num_classes': 4,
            'z_spacing': 1,
        },
        'Kvasir': {
            'Dataset': Kvasir_dataset,
            'volume_path': '../data/Kvasir-SEG',
            'list_dir': './lists/lists_Kvasir',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'ISIC': {
            'Dataset': ISIC_dataset,
            'volume_path': '../data/ISIC2018',
            'list_dir': './lists/lists_ISIC',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'ApproxDA_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'ApproxDA')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_M'+str(args.window_size) if args.window_size!=7 else snapshot_path
    snapshot_path = snapshot_path + '_r'+str(args.rank) if args.rank!=32 else snapshot_path
    snapshot_path = snapshot_path + '_'+args.gate_mode if args.gate_mode!='learn' else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.window_size = args.window_size
    config_vit.rank = args.rank
    config_vit.groups = args.groups
    config_vit.gate_mode = args.gate_mode
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ApproxDATransUNet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    state_dict = torch.load(snapshot, weights_only=False)
    state_dict = {k.replace('.ada_block.', '.approx_block.'): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


