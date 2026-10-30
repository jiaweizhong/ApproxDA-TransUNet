import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jc = metric.binary.jc(pred, gt)
        return dice, hd95, jc
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 1
    else:
        return 0, 0, 0


def calculate_metric_percase_nan(pred, gt, spacing=None):
    """Absent-organ-aware metric (test.py on AMOS). The legacy metric above scores a class that is absent from the
    ground truth as Dice 1 when it is predicted and 0 when it is correctly left out; here such a class returns NaN
    (excluded from the means, nnU-Net convention). A missed organ scores Dice / IoU 0 and HD95 NaN. HD95 is in the
    units of `spacing` and is computed on the bounding box of pred | gt padded by one voxel, which gives the same
    value as the full volume. Also returns the predicted and true voxel counts (hallucinated / missed organs)."""
    pred, gt = pred > 0, gt > 0
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    if n_gt == 0:
        return np.nan, np.nan, np.nan, n_pred, n_gt
    if n_pred == 0:
        return 0.0, np.nan, 0.0, n_pred, n_gt
    box = tuple(slice(max(int(i.min()) - 1, 0), int(i.max()) + 2) for i in np.nonzero(pred | gt))
    p, g = pred[box], gt[box]
    return metric.binary.dc(p, g), metric.binary.hd95(p, g, voxelspacing=spacing), metric.binary.jc(p, g), n_pred, n_gt


def relabel_kidneys_by_side(prediction, left=3, right=4, aorta=1):
    """Synapse post-processing (test.py --lr_post): merge the two kidney labels and give every 3-D connected
    component the side of its mean row relative to the predicted aorta's mean row (in this dataset's orientation
    the left kidney lies at larger row indices than the aorta). Falls back to the image centre without aorta."""
    from scipy.ndimage import label as cc_label

    kid = (prediction == left) | (prediction == right)
    if not kid.any():
        return prediction
    ao_rows = np.nonzero(prediction == aorta)[1]
    mid = ao_rows.mean() if len(ao_rows) else prediction.shape[1] / 2
    comps, n = cc_label(kid)
    idx = np.nonzero(kid)
    ids = comps[idx]
    mean_row = np.bincount(ids, weights=idx[1], minlength=n + 1) / np.maximum(np.bincount(ids, minlength=n + 1), 1)
    out = prediction.copy()
    out[idx] = np.where(mean_row > mid, left, right)[ids]
    return out


def test_single_volume(
    image,
    label,
    net,
    classes,
    patch_size=[256, 256],
    test_save_path=None,
    case=None,
    z_spacing=1,
    lr_post=False,
    metric_mode="legacy",
    spacing=None,
    pred_save_path=None,
    slice_ctx=-1,
    upsample="nearest",
):
    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )
    if len(label.shape) == 3 and slice_ctx >= 1:  # --slice_attn K: stacks of the 2K+1 adjacent slices
        prediction = np.zeros_like(label)
        D, x, y = image.shape
        small = np.stack([zoom(image[z], (patch_size[0] / x, patch_size[1] / y), order=3) if (x, y) != tuple(patch_size)
                          else image[z] for z in range(D)])
        net.eval()
        for ind in range(D):
            idx = np.clip(np.arange(ind - slice_ctx, ind + slice_ctx + 1), 0, D - 1)
            with torch.no_grad():
                out = torch.argmax(net(torch.from_numpy(small[idx]).unsqueeze(0).float().cuda()), dim=1).squeeze(0)
            out = out.cpu().numpy()
            prediction[ind] = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0) if (x, y) != tuple(patch_size) else out
    elif len(label.shape) == 3:  # 3D volume (D, H, W); 2D images have an (H, W) label
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(
                    slice, (patch_size[0] / x, patch_size[1] / y), order=3
                )  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                if upsample != "nearest":  # --upsample bilinear / refine: logits brought to the native slice size
                    if upsample == "refine":
                        native = torch.from_numpy(image[ind]).float()[None, None].cuda()
                        logits = net.point_head.refine(outputs, net.last_dec_feat, native)
                    else:
                        logits = F.interpolate(outputs, size=(x, y), mode="bilinear", align_corners=True)
                    prediction[ind] = torch.argmax(logits, dim=1)[0].cpu().numpy()
                    continue
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        x, y = image.shape[:2]
        if x != patch_size[0] or y != patch_size[1]:
            scale = (patch_size[0] / x, patch_size[1] / y) + (1,) * (image.ndim - 2)
            image_r = zoom(image, scale, order=3)
        else:
            image_r = image
        input = torch.from_numpy(image_r).float()
        # (H, W) grayscale -> (1, 1, H, W); (H, W, 3) RGB -> (1, 3, H, W)
        input = (input.permute(2, 0, 1) if input.ndim == 3 else input.unsqueeze(0)).unsqueeze(0).cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(
                    prediction, (x / patch_size[0], y / patch_size[1]), order=0
                )
    if lr_post and len(label.shape) == 3:
        prediction = relabel_kidneys_by_side(prediction)
    if pred_save_path is not None:  # compact label map for model-independent error analysis
        np.savez_compressed(pred_save_path, prediction=prediction.astype(np.uint8))
    metric_list = []
    for i in range(1, classes):
        if metric_mode == "nan":
            metric_list.append(calculate_metric_percase_nan(prediction == i, label == i, spacing))
        else:
            metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + "/" + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + "/" + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + "/" + case + "_gt.nii.gz")
    return metric_list
