import torch
import torch.nn.functional as F
from tqdm import tqdm

from loss_functions import FocalLoss

def evaluate(net, dataloader, device):

    net.eval()
    num_val_batches = len(dataloader)
    n_classes = 2
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type

    image = image.to(device=device, dtype=torch.float32)
    mask_true = mask_true.to(device=device, dtype=torch.long)

    # true mask를 먼저 interpolate 시키고, 그 다음 one-hot encoding을 진행.
    # mask_true = F.interpolate(mask_true, scale_factor=0.25, mode='nearest')
    mask_true = F.one_hot(mask_true.long().squeeze(1), n_classes).permute(0, 3, 1, 2).float()

    # convert to one-hot format
    if n_classes == 1:
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
        # compute the Dice score
        dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
    else:
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
