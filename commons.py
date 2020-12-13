#alguna wea

from __future__ import division
from bills.models import *
from bills.utils.utils import *
from bills.utils.datasets import *
import os
import sys
import argparse

import cv2
from PIL import Image
import torch
from torch.autograd import Variable

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

def get_detection(img):
	image_folder = "data/samples"
	model_def = "config/yolov3.cfg"
