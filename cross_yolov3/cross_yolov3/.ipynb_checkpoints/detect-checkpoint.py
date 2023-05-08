import os
import tqdm
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from architecture_functions.config_for_arch import CONFIG_ARCH

from general_functions.utils import load_classes, rescale_boxes, non_max_suppression
from general_functions.datasets import ImageFolder
from general_functions.transforms import Resize, DEFAULT_TRANSFORMS


def detect(model, dataloader, output_path, conf_thres, nms_thres):
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
    os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
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


def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """

    image = np.array(Image.open(image_path))
    img = Image.open(image_path)

    detections = rescale_boxes(detections, img_size, image.shape[:2])
    for x1, y1, x2, y2, conf, cls_pred in detections:
        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype(font='./general_functions/ArialMT.ttf', size=np.floor(1.5e-2 * np.shape(img)[1] + 15).astype('int32'))
        label_size = draw.textsize(classes[int(cls_pred)], font)
        text_origin = np.array([x1, y1 - label_size[1]])

        draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
        draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)), fill='red')
        draw.text(tuple(text_origin), str(classes[int(cls_pred)]), fill=(255, 255, 255), font=font)

    # Save generated image with detections
    img.save(os.path.join(output_path, '{}.jpg').format(image_path.split('/')[-1][:-4]))


def _create_detect_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


def run():
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-i", "--images", type=str, default="../data/detrac/test/detect_samples",
                        help="Path to directory with images to inference")
    parser.add_argument("-d", "--data", type=str, default="./config/detrac.data",
                        help="Path to data config file (.data)")
    parser.add_argument("-c", "--classes", type=str, default="../data/detrac.names",
                        help="Path to classes label file (.names)")
    parser.add_argument("-o", "--output", type=str, default="./output/",
                        help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416,
                        help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=12,
                        help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5,
                        help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5,
                        help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names

    # load test dataset
    dataloader = _create_detect_data_loader(args.images, args.batch_size, args.img_size, args.n_cpu)

    # load model
    model_path = CONFIG_ARCH['pruned-model-saving']
    model = torch.load(model_path)

    # detect objects
    img_detections, imgs = detect(model, dataloader, args.output, args.conf_thres, args.nms_thres)

    # draw objects
    for (image_path, detections) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        _draw_and_save_output_image(image_path, detections, args.img_size, args.output, classes)

    print(f"---- Detections were saved to: '{args.output}' ----")


if __name__ == '__main__':
    run()
