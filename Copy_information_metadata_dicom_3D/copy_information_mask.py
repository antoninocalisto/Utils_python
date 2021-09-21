import glob
import numpy as np
import cv2
import pydicom
#from myshow import *
import SimpleITK as sitk
from sklearn.preprocessing import MinMaxScaler
import subprocess, os
from PIL import Image, ImageOps
import fnmatch


name_volume = "path_1/Volume/_1.mha"

name_mask = "path_1/Mask/_mask_temp_1.nii"

volumn = sitk.ReadImage(name_volume)
mask = sitk.ReadImage(name_mask) 

mask.CopyInformation(volumn)


sitk.WriteImage(mask, '_mask_temp_1_new.nii')