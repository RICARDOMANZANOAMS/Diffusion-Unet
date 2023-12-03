from PIL import Image
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch

class Read_file:
    def __init__(self, image_path):
        self.image_path=image_path


    def show_image(self):
        """
        Function to show image.
        Use path passed in the class
        """
        img=Image.open(self.image_path)
        plt.imshow(img)
        plt.axis('off')  # Turn off axis labels
        plt.show()  
        return img
    
    def read_image(self):

        """
        This function is used to read the image and transform it to tensor
        Use path passed in the class
        """
        img=self.show_image()
        torch_img=transforms.ToTensor()(img)
        print(torch_img)



if __name__ == "__main__": 
    image=Read_file('./data/Lion.jpg')
    image.read_image()
