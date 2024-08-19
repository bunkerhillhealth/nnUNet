import torch


def process_output_target_for_val(
    has_regions: bool,
    has_ignore_label: bool,
    ignore_label: int,
    output: torch.Tensor,
    target: torch.Tensor,
):
    """
    Function processed model outputand taarget to make it read for eval. This includes:
    - Applying sigmoid
    - Processing appropriately in case of region ased training and label ignoring
    """
    if has_regions:
        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
    else:
        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.float32
        )
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

    if has_ignore_label:
        if not has_regions:
            mask = (target != ignore_label).float()
            # CAREFUL that you don't rely on target after this line!
            target[target == ignore_label] = 0
        else:
            mask = 1 - target[:, -1:]
            # CAREFUL that you don't rely on target after this line!
            target = target[:, :-1]
    else:
        mask = None

    return predicted_segmentation_onehot, target, mask
