from ImageUtilities import ImageUtilities
from Transformations import Transformations
import torch

# Add device import torch
if torch .cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
device = torch.device("cuda:1")
print(device)

img_util=ImageUtilities()
img_tensor=img_util.read_image('./data/Lion.jpg')
img=img_util.plot_tensor_image(img_tensor)  #plot image from tensor

trans_obj=Transformations()
img_resize=trans_obj.resize(img_tensor,50, 50)
img=img_util.plot_tensor_image(img_resize)  #plot image from tensor

a=trans_obj.min_max_transform(img_resize, apply_resize=False)
print(a)
b=trans_obj.reverse_min_max_transform(a)


img=img_util.plot_tensor_image(b)  #plot image from tensor
# Preprocess_ins=Transformations(tensor)
# a=Preprocess_ins.transform(apply_resize=True, resize_height=50, resize_width=50)
# print(a)
# b=Preprocess_ins.reverse_transform()
# print(b)
