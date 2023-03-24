import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.select_dataset import select_dataset
from models.select_model import select_model
from utils.metrics import pixel_level_metrics_per_image, image_level_metrics, pixel_level_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    model_name = 'mantranet'
    inference_dataset_name = 'casia1'
    
    predicted_save_folder = f'./res/{model_name}/{inference_dataset_name}'
    evaluate_result_file_path = f'./res/{model_name}/{inference_dataset_name}_evaluate_result.txt'
    os.makedirs(predicted_save_folder, exist_ok=True)
    evaluate_result_file = open(evaluate_result_file_path, 'w')

    model = select_model(model_name=model_name)
    model = nn.DataParallel(model).cuda()
    model.eval()

    infer_dataset = select_dataset(dataset_name=inference_dataset_name)
    infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=4)
    infer_num = len(infer_dataset)
    print(f'Using {infer_num} images for inference.')

    image_scores, image_labels = [], []
    pixel_aucs, pixel_precision_th05s, pixel_recall_th05s, pixel_f1_th05s = [], [], [], []

    with torch.no_grad():
        train_bar = tqdm(infer_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            inputs, masks, image_paths, mask_paths = data
            inputs = inputs.cuda()

            image_path, mask_path = image_paths[0], mask_paths[0]

            outputs = model(inputs)

            output = outputs.squeeze(dim=0).squeeze(dim=0)
            # output = torch.sigmoid(output)
            output = output.cpu().numpy()

            mask = masks.squeeze(dim=0).squeeze(dim=0).numpy()

            label = np.max(mask)
            score = np.max(output)
            
            image_scores.append(score)
            image_labels.append(label)

            if label > 0:
                pixel_auc, pixel_precision_th05, pixel_recall_th05, pixel_f1_th05 = pixel_level_metrics_per_image(output, mask)
                pixel_aucs.append(pixel_auc)
                pixel_precision_th05s.append(pixel_precision_th05)
                pixel_recall_th05s.append(pixel_recall_th05)
                pixel_f1_th05s.append(pixel_f1_th05)

    image_acc, image_auc, image_sensitivity, image_specificity, image_f1 = image_level_metrics(image_scores, image_labels)
    pixel_auc, pixel_precision_th05, pixel_recall_th05, pixel_f1_th05 = pixel_level_metrics(pixel_aucs, pixel_precision_th05s, pixel_recall_th05s, pixel_f1_th05s)

    evaluate_result_file.write('=' * 32 + '\n')
    evaluate_result_file.write(f'image_auc: {image_auc*100:.2f}, image_sensitivity: {image_sensitivity*100:.2f}, image_specificity: {image_specificity*100:.2f}, image_f1: {image_f1*100:.2f}\n')
    evaluate_result_file.write(f'pixel_auc: {pixel_auc*100:.2f}, pixel_precision_th05: {pixel_precision_th05*100:.2f}, pixel_recall_th05: {pixel_recall_th05*100:.2f}, pixel_f1_th05: {pixel_f1_th05*100:.2f}\n')
    evaluate_result_file.write('=' * 32 + '\n')

    print(f'image_auc: {image_auc*100:.2f}, image_sensitivity: {image_sensitivity*100:.2f}, image_specificity: {image_specificity*100:.2f}, image_f1: {image_f1*100:.2f}')
    print(f'pixel_auc: {pixel_auc*100:.2f}, pixel_precision_th05: {pixel_precision_th05*100:.2f}, pixel_recall_th05: {pixel_recall_th05*100:.2f}, pixel_f1_th05: {pixel_f1_th05*100:.2f}')
        
    evaluate_result_file.close()

