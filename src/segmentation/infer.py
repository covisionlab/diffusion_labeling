
import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import time
import argparse
from tqdm import tqdm
import torch
from lightning.fabric import Fabric

from src.segmentation.core.pytorch_ext_local import F1MeasureMultich, F1Measure, ConfusionMatrix
from src.segmentation.core.model_factory import model_factory
from src.segmentation.core.data_factory import data_factory
from src.segmentation.core.utils import parse_opts, visualize_predictions, visualize_batch, visualize_infer_batch



def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ###### Logging #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/segmentation/configs/inference.yaml')
    opt, opt_dict = parse_opts(parser.parse_args().config)

    # Initialize fabric
    fabric = Fabric(accelerator='gpu', devices=[0])
    fabric.seed_everything(opt.main.seed)
    fabric.launch()
    opt.device = fabric.device

    ######  Setup  #######
    model = model_factory(opt)
    train_dset, val_dset, train_loader, val_loader = data_factory(opt)
    val_loader = fabric.setup_dataloaders(val_loader)
    model = fabric.setup(model)

    ######  val  #######
    # run inference and compute mean time for whole validation
    with torch.inference_mode(True):
        model.eval()
        samples_seen = 0
        measure = F1Measure()
        confusion = ConfusionMatrix()
        mu_iou = 0

        # bnorm = 1 always
        # time
        start = time.time()
        for batch in (pbar := tqdm(val_loader)):
            
            x, mask = batch
            x = x.to(fabric.device)
            mask = mask.to(fabric.device)

            out = model(x)

            # visualize_infer_batch(x, out>opt.hyp.threshold, mask, val_dset.unnorm, f"viz/infer_sdf/output_{samples_seen}.png")

            # metrics
            mu_f1 = measure.update(out>opt.hyp.threshold, mask)
            confusion.update(out>opt.hyp.threshold, mask)
            tp, fp, fn, tn = confusion.get()
            mu_iou += tp / (tp + fp + fn)

            samples_seen += x.shape[0]


            # log   
            pbar.set_description(f"[EPOCH {samples_seen:03} val] m_bf1:{mu_f1:.4f} m_iou:{mu_iou/samples_seen:.4f}")


        # log
        confusion.pretty_print()


        # time
        end = time.time()
        print(f"Time taken for whole validation: {end-start} seconds")
        print(f"Time taken for single image: {(end-start)/samples_seen} seconds")


if __name__ == '__main__':
    main()

# 0.5
# sdf    TP: 83789, FP: 247142, FN: 4860, TN: 90316881  0.3994 0.2510
# layout TP: 72531, FP: 58697, FN: 16118, TN: 90505326  0.6597 0.4963

# 0.75
# sdf    TP: 82057, FP: 179331, FN: 6592, TN: 90384692 0.4688  0.3061  
# layout TP: 68442, FP: 39669, FN: 20207, TN: 90524354 0.6957  0.5278


# 0.90
# sdf TP: 79840, FP: 131056, FN: 8809, TN: 90432967    0.5331 0.3627
# layout TP: 64406, FP: 27662, FN: 24243, TN: 90536361 0.7128 0.5384