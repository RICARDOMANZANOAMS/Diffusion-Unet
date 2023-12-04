import torchvision.transforms as transforms
import torch
import numpy as np
class Transformations:
    def __init__(self):
        self.tensor=None
       

    def resize(self,tensor, height, width):
        tensor_resize = transforms.Resize((height, width))(tensor)
        return tensor_resize

    def min_max_transform(self,tensor, apply_resize=True, resize_height=50, resize_width=50):
        """
        Function to transform a tensor image to [-1 1]: scaling and normalizing between -1 and 1
        """
        if apply_resize:
            tensor_resize = self.resize(resize_height, resize_width)
        else:
            tensor_resize = tensor

        # Apply normalization to scale data between [-1, 1]
        norm_image = transforms.functional.normalize(tensor_resize, mean=0.5, std=0.5)

        print(norm_image)
        return norm_image

    def reverse_min_max_transform(self,tensor):
        """"
        This function takes -1 and 1 tensor and transforms to 0 to 255 image
        """
        scale_tensor=transforms.Lambda(lambda t: (t + 1) / 2)(tensor) # Scale data between [0,1]
        change_channels=transforms.Lambda(lambda t: t.permute(1, 2, 0))(scale_tensor) # CHW to HWC
        transform_real_img=transforms.Lambda(lambda t: t * 255.)( change_channels) # Scale data between [0.,255.]
        int_img=transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8))(transform_real_img) # Convert into an uint8 numpy array
        
        return int_img
       




        
        