import torch
from torch import autocast, bfloat16
from tqdm import tqdm
from src.utils.metrics import compute as compute_metrics


def validate_or_test(model, data_loader, loss_fn, device, header):
    model.eval()

    all_preds = []
    all_targets = []

    running_loss = 0.0
    total_samples = 0

    # with mixed precision
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=header, leave=False)
        for targets, ct_batch, pet_batch in progress_bar:
            targets = targets.to(device, non_blocking=True)
            ct_batch = ct_batch.to(device, non_blocking=True)
            pet_batch = pet_batch.to(device, non_blocking=True)

            with autocast(device.type, dtype=bfloat16):
                outputs = model(ct_batch, pet_batch)
                losses = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            batch_size = targets.size(0)
            running_loss += losses.item() * batch_size
            total_samples += batch_size
            progress_bar.set_postfix(loss=losses.item())

    torch.cuda.empty_cache()

    return compute_metrics(all_targets, all_preds, running_loss, total_samples)
