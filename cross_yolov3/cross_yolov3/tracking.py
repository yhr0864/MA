import numpy as np
import torch
import argparse
from PIL import Image
from detect import detect, _create_detect_data_loader
from general_functions.sort import *
from architecture_functions.config_for_arch import CONFIG_ARCH
from general_functions.utils import rescale_boxes


parser = argparse.ArgumentParser(description="tracing")
parser.add_argument("-i", "--images", type=str, default="../data/detrac/test/tracking_seq(only_for_test)",
                    help="Path to directory with images to inference")
parser.add_argument("-b", "--batch_size", type=int, default=8,
                    help="Size of each image batch")
parser.add_argument("--img_size", type=int, default=416,
                    help="Size of each image dimension for yolo")
parser.add_argument("--n_cpu", type=int, default=12,
                    help="Number of cpu threads to use during batch generation")
parser.add_argument("-o", "--output", type=str, default="./output_tracking/",
                    help="Path to output directory")
parser.add_argument("--conf_thres", type=float, default=0.5,
                    help="Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5,
                    help="IOU threshold for non-maximum suppression")

def track():
    # create instance of SORT
    mot_tracker = Sort()

    # get detections
    model_path = CONFIG_ARCH['pruned-model-saving']
    model = torch.load(model_path)

    dataloader = _create_detect_data_loader(args.images, args.batch_size, args.img_size, args.n_cpu)

    img_detections, imgs = detect(model, dataloader, args.output, args.conf_thres, args.nms_thres)

    image = np.array(Image.open(args.images))
    for (image_path, detections) in zip(imgs, img_detections): # for each frame
        detections = rescale_boxes(detections, args.img_size, image.shape[:2])
        detections = np.array(detections)[:, :5]

        # update SORT
        track_bbs_ids = mot_tracker.update(detections)

    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    print(track_bbs_ids)

if __name__ == "__main__":
    track()