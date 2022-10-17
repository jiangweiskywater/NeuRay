import argparse
import os
import lpips

import torch
from skimage.io import imread
from tqdm import tqdm
import numpy as np

from utils.base_utils import color_map_forward
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

class Evaluator:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='vgg').cuda().eval()
        # self.loss_fn_alex = lpips.LPIPS(net='alex').cuda().eval()

    def eval_metrics_img(self,gt_img, pr_img):
        gt_img = color_map_forward(gt_img)
        pr_img = color_map_forward(pr_img)
        psnr = tf.image.psnr(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        ssim = tf.image.ssim(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        with torch.no_grad():
            gt_img_th = torch.from_numpy(gt_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            pr_img_th = torch.from_numpy(pr_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            score = float(self.loss_fn_alex(gt_img_th, pr_img_th).flatten()[0].cpu().numpy())
        return float(psnr), float(ssim), score

    def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, ignore_label):
        if (true_labels == ignore_label).all():
            return [0]*4

        true_labels = true_labels.flatten()
        predicted_labels = predicted_labels.flatten()
        valid_pix_ids = true_labels!=ignore_label
        predicted_labels = predicted_labels[valid_pix_ids] 
        true_labels = true_labels[valid_pix_ids]

        conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
        norm_conf_mat = np.transpose(
            np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

        missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
        exsiting_class_mask = ~ missing_class_mask

        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        ious = np.zeros(number_classes)
        for class_id in range(number_classes):
            ious[class_id] = (conf_mat[class_id, class_id] / (
                    np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                    conf_mat[class_id, class_id])) 
        miou = nanmean(ious)
        miou_valid_class = np.mean(ious[exsiting_class_mask])
        return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious


    def eval(self, dir_gt, dir_pr, dir_gt_sem):
        results=[]
        num = len(os.listdir(dir_gt))
        for k in tqdm(range(0, num)):
            pr_img = imread(f'{dir_pr}/{k}-nr_fine.jpg')
            gt_img = imread(f'{dir_gt}/{k}.jpg')

            psnr, ssim, lpips_score = self.eval_metrics_img(gt_img, pr_img)
            results.append([psnr,ssim,lpips_score])
        psnr, ssim, lpips_score = np.mean(np.asarray(results),0)

        msg=f'psnr {psnr:.4f} ssim {ssim:.4f} lpips {lpips_score:.4f}'
        print(msg)

        sem_results=[]
        num = len(os.listdir(dir_gt_sem))
        for k in tqdm(range(0, num)):
            pr_sem = imread(f'{dir_pr}/{k}-sem_nr_fine.jpg')
            gt_sem = imread(f'{dir_gt_sem}/{k}_sem.jpg')

            miou, miou_valid_class, total_accuracy, class_average_accuracy, ious = self.calculate_segmentation_metrics(gt_sem, pr_sem, 40, -1)
            sem_results.append([miou, miou_valid_class, total_accuracy, class_average_accuracy])
        
        miou, miou_valid_class, total_accuracy, class_average_accuracy= np.mean(np.asarray(sem_results),0)

        msg2=f'miou {miou:.4f} miou_valid_class {miou_valid_class:.4f} total_accuracy {total_accuracy:.4f} class_average_accuracy {class_average_accuracy:.4f}'
        print(msg2)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_gt', type=str, default='data/render/fern/gt')
    parser.add_argument('--dir_pr', type=str, default='data/render/fern/neuray_gen_depth-pretrain-eval')
    flags = parser.parse_args()
    evaluator = Evaluator()
    evaluator.eval(flags.dir_gt, flags.dir_pr)
