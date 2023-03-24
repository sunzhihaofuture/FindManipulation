import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.select_model import select_model

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    model_name = 'mantranet'
    image_path = './pics/demo_image.png'
    predicted_save_path = image_path.replace('.png', f'_{model_name}_pred.png').replace('.jpg', f'_{model_name}_pred.jpg')

    model = select_model(model_name=model_name)
    model = nn.DataParallel(model).cuda()
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = torch.Tensor(np.array(image))
    inputs = inputs.permute(2, 0, 1).unsqueeze(dim=0).cuda()

    with torch.no_grad():
        outputs = model(inputs)

    plt.figure(figsize=(16, 9))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow((outputs[0][0]).cpu().detach(), cmap='gray')
    plt.title('Predicted forgery mask')
    
    plt.subplot(1, 3, 3)
    plt.imshow((outputs[0][0].cpu().detach().unsqueeze(2) > 0.5) * torch.tensor(image))
    plt.title('Suspicious regions detected')

    plt.savefig(predicted_save_path)
    plt.show()
        
