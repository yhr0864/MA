import gc
import time
import timeit

import tqdm
import torch
import argparse
import numpy as np
from thop import profile

from torch.autograd import Variable

from supernet_main_file import _create_test_data_loader
from architecture_functions.config_for_arch import CONFIG_ARCH
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

from general_functions.utils import (parse_data_config, ap_per_class, xywh2xyxy,
                                     non_max_suppression, get_batch_statistics)


def get_lat_and_macs(model):
    input_sample = torch.randn(1, 3, 416, 416)
    cnt_of_runs = 50
    globals()['op'], globals()['input_sample'] = model, input_sample
    total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()",
                               globals=globals(), number=cnt_of_runs)
    avg_time = total_time / cnt_of_runs * 1e6
    macs, params = profile(model, inputs=(input_sample,))
    print("latency: ", avg_time)
    print("MACs: ", macs)
    print("params: ", params)
    
    
def print_test_logging(start_time, metrics_output):
    precision, recall, AP, f1, ap_class = metrics_output
    class_names = ["car", "van", "bus", "others"]
    print("Index", "Class", "AP")
    for i, c in enumerate(ap_class):
        print(c, class_names[c], "%.5f" % AP[i])
    
    print("test : Final Precision {:.5%}, Time {:.2f}".format(AP.mean().item(),
                                                               time.time() - start_time))


def _test(loader, model, img_size, args):
    model.eval()
    start_time = time.time()
    labels = []
    sample_metrics = []

    for _, images, targets in tqdm.tqdm(loader, desc="testing"):       
        images, targets = images.cuda(), targets.cuda()
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        images = Variable(images, requires_grad=False)

        with torch.no_grad():
            outs = model(images)
            outs = non_max_suppression(outs, conf_thres=args.conf_thres,
                                       iou_thres=args.nms_thres)

        sample_metrics += get_batch_statistics(outs, targets.cpu(),
                                               iou_threshold=args.iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    if metrics_output is not None:
        print_test_logging(start_time=start_time, metrics_output=metrics_output)
    else:
        print(" mAP not measured (no detections found by model) for Time {:.2f}".format(
              time.time() - start_time))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("test")
    parser.add_argument("-d", "--data", type=str, default="./config/detrac.data",
                        help="Path to data config file (.data)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size to test")
    parser.add_argument("--n_cpu", type=int, default=12,
                        help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()

    data_config = parse_data_config(args.data)
    test_path = data_config["test"]

    test_loader = _create_test_data_loader(test_path,
                                           args.batch_size,
                                           CONFIG_SUPERNET['dataloading']['img_size'],
                                           args.n_cpu)

    model_path = CONFIG_ARCH['train_settings']['path_to_save_model']
    # model_path = CONFIG_ARCH['pruned-model-saving']
    model = torch.load(model_path)

    _test(test_loader, model, CONFIG_SUPERNET['dataloading']['img_size'], args)
    
    get_lat_and_macs(model.cpu())