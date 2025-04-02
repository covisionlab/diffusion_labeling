import json

import torch
import numpy as np
from tqdm import tqdm
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from utils.analog_bits import bits_to_decimal, decimal_to_bits


def get_random_bboxes(opt, dataset):
    all_bboxes = []
    all_imgs = []
    for b in range(opt.n_viz_images):
        # get a random image from the dataset
        idx = torch.randint(0, len(dataset), (1,))
        _, img, _, bboxes, _, _ = dataset[idx]
        img = (img + 1) / 2
        all_imgs.append(img)
        all_bboxes.append(torch.tensor(bboxes))
    
    return torch.tensor(np.array(all_imgs)), all_bboxes


def get_random_bboxes_sdf_and_masks(opt, dataset):
    all_images = []
    all_bbox_sdf = []
    all_class_map = []
    all_bboxes = []
    for b in range(opt.n_viz_images):
        # get a random image from the dataset
        idx = torch.randint(0, len(dataset), (1,))
        _, img, _, bboxes, bbox_sdf, class_map = dataset[idx]
        all_images.append((img + 1) / 2)
        all_bbox_sdf.append(bbox_sdf)
        all_class_map.append(class_map)
        all_bboxes.append(torch.tensor(bboxes))
    
    return torch.tensor(np.array(all_images)), torch.tensor(np.array(all_bbox_sdf)), torch.tensor(np.array(all_class_map)), all_bboxes


class DiffPipeLayout(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        
        self.register_modules(model=model, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(self, opt, dataset, device):
        ###### sampling ######
        ######################
        # initial noise
        x = torch.randn(opt.n_viz_images, opt.out_channels, opt.image_height, opt.image_width).to(device)
        
        # get random bounding boxes
        layout_length = json.loads(opt.network_config)['layout_encoder']['layout_length']
        num_classes_for_layout_object = json.loads(opt.network_config)['layout_encoder']['num_classes_for_layout_object']
        random_imgs, random_bboxes = get_random_bboxes(opt, dataset)
        
        # get bbox and class for layout encoder
        obj_bbox = torch.zeros([opt.n_viz_images, layout_length, 4])
        obj_bbox[:, 0] = torch.tensor([[0, 0, 1, 1]]).repeat(opt.n_viz_images, 1)
        obj_class = torch.LongTensor(opt.n_viz_images, layout_length).fill_(num_classes_for_layout_object - 1)
        obj_class[:, 0] = torch.LongTensor(opt.n_viz_images).fill_(0)
        for n, bbox in enumerate(random_bboxes):
            curr_classes = bbox[:, 0]
            curr_bboxes = bbox[:, 1:]
            obj_bbox[n, 1:1+len(curr_bboxes)] = curr_bboxes
            obj_class[n, 1:1+len(curr_classes)] = curr_classes
        obj_bbox = obj_bbox.to(device)
        obj_class = obj_class.to(device)
        
        # For each sampling timestep
        for t in tqdm(self.scheduler.timesteps):
            timestep = torch.tensor([t] * opt.n_viz_images).to(device)
            with torch.no_grad():
                noise_pred = self.model(x, timestep, obj_class, obj_bbox)[0]

            # Update the sample with the predicted noise
            x = self.scheduler.step(noise_pred, t.item(), x).prev_sample

        ##### postprocessing #####
        ##########################
        # Clip output to range [-1, 1]
        x = x.cpu().clip(-1, 1)
        
        # get image and normalize to [0, 1]
        img = x[:, :3, :, :]
        img = (img + 1) / 2
        
        # get mask and convert to decimal
        mask = x[:, 3:, :, :]
        mask = bits_to_decimal(mask, bits=int(np.ceil(np.log2(opt.num_classes))))

        return img, mask, random_bboxes, random_imgs


class DiffPipeLayoutTest(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        
        self.register_modules(model=model, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(self, opt, batch, device):
        # initial noise
        x = torch.randn(opt.batch_size, opt.out_channels, opt.image_height, opt.image_width).to(device)
        
        # get layout encoder hyperparams
        layout_length = json.loads(opt.network_config)['layout_encoder']['layout_length']
        num_classes_for_layout_object = json.loads(opt.network_config)['layout_encoder']['num_classes_for_layout_object']
        
        # get bbox and class for layout encoder
        obj_bbox = torch.zeros([opt.batch_size, layout_length, 4])
        obj_bbox[:, 0] = torch.tensor([[0, 0, 1, 1]]).repeat(opt.batch_size, 1)
        obj_class = torch.LongTensor(opt.batch_size, layout_length).fill_(num_classes_for_layout_object - 1)
        obj_class[:, 0] = torch.LongTensor(opt.batch_size).fill_(0)
        for n, bbox in enumerate(batch['bbox']):
            curr_classes = bbox[:, 0]
            curr_bboxes = bbox[:, 1:]
            obj_bbox[n, 1:1+len(curr_bboxes)] = curr_bboxes
            obj_class[n, 1:1+len(curr_classes)] = curr_classes
        obj_bbox = obj_bbox.to(device)
        obj_class = obj_class.to(device)
        
        # For each sampling timestep
        for t in tqdm(self.scheduler.timesteps):
            timestep = torch.tensor([t] * opt.batch_size).to(device)
            with torch.no_grad():
                noise_pred = self.model(x, timestep, obj_class, obj_bbox)[0]

            # Update the sample with the predicted noise
            x = self.scheduler.step(noise_pred, t.item(), x).prev_sample

        ##### postprocessing #####
        ##########################
        # Clip output to range [-1, 1]
        x = x.cpu().clip(-1, 1)
        
        # get image and normalize to [0, 1]
        img = x[:, :3, :, :]
        img = (img + 1) / 2
        
        # get mask and convert to decimal
        mask = x[:, 3:, :, :]
        mask = bits_to_decimal(mask, bits=int(np.ceil(np.log2(num_classes_for_layout_object - 1))))

        return img, mask


class DiffPipeOurs(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        
        self.register_modules(model=model, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(self, opt, dataset, device, test=False, selected_ids=None):
        ###### sampling ######
        ######################
        # initial noise
        x = torch.randn(opt.n_viz_images, opt.out_channels, opt.image_height, opt.image_width).to(device)
        
        # get random bounding boxes and masks
        random_imgs, random_bbox_sdf, random_class_map, random_bboxes = get_random_bboxes_sdf_and_masks(opt, dataset)
        random_bbox_sdf = random_bbox_sdf.to(device)
        random_class_map = random_class_map.to(device)
        random_class_map = decimal_to_bits(random_class_map, bits=int(np.ceil(np.log2(opt.num_classes))))
                
        # For each sampling timestep
        for t in tqdm(self.scheduler.timesteps):
            timestep = torch.tensor([t] * opt.n_viz_images).to(device)
            with torch.no_grad():
                noise_pred = self.model(torch.cat((x, random_bbox_sdf, random_class_map), dim=1), timestep)[0]

            # Update the sample with the predicted noise
            x = self.scheduler.step(noise_pred, t.item(), x).prev_sample

        ##### postprocessing #####
        ##########################
        # Clip output to range [-1, 1]
        x = x.cpu().clip(-1, 1)
        
        # get image and normalize to [0, 1]
        img = x[:, :3, :, :]
        img = (img + 1) / 2
        
        # get mask and convert to decimal
        mask = x[:, 3:, :, :]
        mask = bits_to_decimal(mask, bits=int(np.ceil(np.log2(opt.num_classes))))

        return img, mask, random_bboxes, random_imgs


class DiffPipeOursTest(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        
        self.register_modules(model=model, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(self, opt, batch, device):
        bs = len(batch['bbox'])
        
        # initial noise
        x = torch.randn(bs, opt.out_channels, opt.image_height, opt.image_width).to(device)
        
        bbox_sdf = batch['sdf_map'].to(device)
        class_map = batch['class_map'].to(device)
        class_map = decimal_to_bits(class_map, bits=int(np.ceil(np.log2(opt.num_classes))))
                
        # For each sampling timestep
        for t in tqdm(self.scheduler.timesteps):
            timestep = torch.tensor([t] * bs).to(device)
            with torch.no_grad():
                noise_pred = self.model(torch.cat((x, bbox_sdf, class_map), dim=1), timestep)[0]

            # Update the sample with the predicted noise
            x = self.scheduler.step(noise_pred, t.item(), x).prev_sample

        ##### postprocessing #####
        ##########################
        # Clip output to range [-1, 1]
        x = x.cpu().clip(-1, 1)
        
        # get image and normalize to [0, 1]
        img = x[:, :3, :, :]
        img = (img + 1) / 2
        
        # get mask and convert to decimal
        mask = x[:, 3:, :, :]
        mask = bits_to_decimal(mask, bits=int(np.ceil(np.log2(opt.num_classes))))

        return img, mask