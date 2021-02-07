import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image, title=None):
    '''
    Plots input image
    
    takes as input image represented by torch tensor (might take 
    tuple taking from torch dataset like this: dataset[i]) or np.ndarray.
    Works with both channels first or last and extra 0 dimension.
    
    title - regular matplotlib title
    
    '''   
    plt.title(title)
    if type(image) == tuple and type(image[0]) == torch.Tensor:
        image = image[0]
        
    if len(image.shape) == 4:
        if type(image) == torch.Tensor:
            image = image.squeeze(0)
        if type(image) == np.ndarray:
            image = image[0]
    
    channels_first = False
    if image.shape[0] < image.shape[1]:
        channels_first = True
    
    if channels_first:
        if type(image) == torch.Tensor:
            plt.imshow(image.permute(1, 2, 0))
            plt.show()
        if type(image) == np.ndarray:
            plt.imshow(torch.from_numpy(image).permute(1,2,0))
            plt.show()
    else:
        plt.imshow(image)
        plt.show()

