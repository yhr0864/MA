import numpy as np
import torch
import argparse
from tqdm import tqdm
from PIL import Image

from torch.autograd import Variable
from detect import _create_detect_data_loader
from general_functions.sort import *
from architecture_functions.config_for_arch import CONFIG_ARCH
from general_functions.utils import rescale_boxes, non_max_suppression


def detect(model, dataloader, conf_thres, nms_thres):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    # Create output directory, if missing
    # os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    for (img_paths, input_imgs) in dataloader:
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)
    return img_detections, imgs

parser = argparse.ArgumentParser(description="tracing")
parser.add_argument("-i", "--images", type=str, default="/pfs/data5/home/kit/tm/px6680/haoran/data/detrac/test/inference_seq/",
                    help="Path to directory with images to inference")
parser.add_argument("-b", "--batch_size", type=int, default=1,
                    help="Size of each image batch")
parser.add_argument("--img_size", type=int, default=416,
                    help="Size of each image dimension for yolo")
parser.add_argument("--n_cpu", type=int, default=12,
                    help="Number of cpu threads to use during batch generation")
parser.add_argument("-o", "--output", type=str, default="./output_tracking/",
                    help="Path to output directory")
parser.add_argument("--conf_thres", type=float, default=0.5,
                    help="Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4,
                    help="IOU threshold for non-maximum suppression")
args = parser.parse_args()

def track():
    # create instance of SORT
    mot_tracker = Sort()

    # get detections
    model_path = CONFIG_ARCH['train_settings']['path_to_save_model']
    model = torch.load(model_path)
    
    # img seq
    seq_folder = args.images
    seq_list = os.listdir(seq_folder)
    seq_list.sort()
    
    id_list = []
    with tqdm(total=len(seq_list)) as pbar:
        for i, seq in enumerate(seq_list):
            
            if i == 40:
                image_folder = seq_folder + seq
                dataloader = _create_detect_data_loader(image_folder, args.batch_size, args.img_size, args.n_cpu)

                img_detections, imgs = detect(model, dataloader, args.conf_thres, args.nms_thres)

                first_img = glob.glob("%s/*.*" % image_folder)[0]
                image = np.array(Image.open(first_img))

                for (image_path, detections) in zip(imgs, img_detections): # for each frame
                    detections = rescale_boxes(detections, args.img_size, image.shape[:2])
                    detections = np.array(detections)[:, :5]

                    # update SORT
                    track_bbs_ids = mot_tracker.update(detections)

                # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
                if len(track_bbs_ids) != 0:
                    id_list.append(max(track_bbs_ids[:, -1]))
                else:
                    id_list.append(0)

            pbar.update(1)
            
    
    print(id_list)


if __name__ == "__main__":
    track()