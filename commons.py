from __future__ import division

from models import *
from utils.utils import *   
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import io

import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

def get_model():
	chekpoint='bills/checkpoints/yolov3_ckpt_97.pth'
	model = models.densenet121(pretraiend=True)
	model.classifier=nn.Linear(1024, 102)
	model.load_state_dict(torch.load(
		chekpoint, map_location='cpu'),strict = False)
	model.eval()
	return model

def get_tensor(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(255),
										transforms.CemterCrop(224),
										transforms.ToTensor(),
										transforms.Normalize(
											[0.485, 0.456, 0.406],
											[0.229, 0.224, 0.225])])
	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image)

def get_detection(image):
	img_paths = image
	model_def = "config/yolov3-custom.cfg"
	weights_path = "weights/yolov3.weights"
	class_path = "data/custom/classes.names"
	conf_thres = 0.85
	nms_thres = 0.4
	batch_size = 1
	n_cpu = 1
	img_size = 416
	checkpoint_model = "checkpoint/yolov3_ckpt_97.pth"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = Darknet(model_def, img_size=img_size).to(device)
	model.load_darknet_weights(weights_path)
	model.eval()  # Set in evaluation mode

	dataloader = DataLoader(
        ImageFolder(img_paths, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )

	classes = load_classes(opt.class_path)  # Extracts class labels from file
	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        print("INPUT_IMGS-----", input_imgs)
        input_imgs = Variable(input_imgs.type(Tensor))
        

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        #Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

        return
