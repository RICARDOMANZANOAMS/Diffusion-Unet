from PIL import Image
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch

class ImageUtilities:
    def __init__(self):
        self.tensor=None


    def plot_tensor_image(self,tensor_img):
        """
        Function to show image.
        It is necessary to pass the tensor that contains the image
        """
        pil_img=transforms.ToPILImage()(tensor_img) # Convert to PIL image
        plt.imshow(pil_img)
        plt.axis('off')  # Turn off axis labels
        plt.show()  
       
    
    def read_image(self,image_path):

        """
        This function is used to read the image and transform it to tensor
        Pass the image path 
        """
        img=Image.open(image_path)
        tensor=transforms.ToTensor()(img)  #It reads the image and transform the values of pixels between 0 and 1
        self.tensor=tensor
        return tensor



