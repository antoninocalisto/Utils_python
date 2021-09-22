
#from myshow import *
import SimpleITK as sitk



name_volume = "path_1/Volume/_1.mha"

name_mask = "path_1/Mask/_mask_temp_1.nii"

volumn = sitk.ReadImage(name_volume)
mask = sitk.ReadImage(name_mask) 

mask.CopyInformation(volumn)


sitk.WriteImage(mask, '_mask_temp_1_new.nii')