from . import pspnet
from . import unet
from . import segnet
from . import fcn
model_from_name = {}


model_from_name["fcn_8"] = fcn.fcn_8
model_from_name["fcn_32"] = fcn.fcn_32
model_from_name["fcn_8_vgg"] = fcn.fcn_8_vgg
model_from_name["fcn_32_vgg"] = fcn.fcn_32_vgg
model_from_name["fcn_8_resnet50"] = fcn.fcn_8_resnet50
model_from_name["fcn_32_resnet50"] = fcn.fcn_32_resnet50
model_from_name["fcn_8_mobilenet"] = fcn.fcn_8_mobilenet
model_from_name["fcn_32_mobilenet"] = fcn.fcn_32_mobilenet


model_from_name["pspnet"] = pspnet.pspnet
model_from_name["vgg_pspnet"] = pspnet.vgg_pspnet
model_from_name["resnet50_pspnet"] = pspnet.resnet50_pspnet

model_from_name["vgg_pspnet"] = pspnet.vgg_pspnet
model_from_name["resnet50_pspnet"] = pspnet.resnet50_pspnet

model_from_name["pspnet_50"] = pspnet.pspnet_50
model_from_name["pspnet_101"] = pspnet.pspnet_101


# model_from_name["mobilenet_pspnet"] = pspnet.mobilenet_pspnet


model_from_name["unet_mini"] = unet.unet_mini
model_from_name["unet"] = unet.unet
model_from_name["vgg_unet"] = unet.vgg_unet
model_from_name["resnet50_unet"] = unet.resnet50_unet
model_from_name["mobilenet_unet"] = unet.mobilenet_unet


model_from_name["segnet"] = segnet.segnet
model_from_name["vgg_segnet"] = segnet.vgg_segnet
model_from_name["resnet50_segnet"] = segnet.resnet50_segnet
model_from_name["mobilenet_segnet"] = segnet.mobilenet_segnet
