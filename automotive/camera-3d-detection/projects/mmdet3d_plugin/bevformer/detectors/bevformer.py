# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import pdb


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        # import pdb
        # pdb.set_trace()
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            # pdb.set_trace()
            # torch.onnx.export(self.img_neck, img_feats[0], 'bevformer_small_epoch_24_conv2d_neck.onnx', verbose=False, opset_version=14, dynamic_axes=None)
            img_feats = self.img_neck(img_feats[0])

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(
                    int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, img=None, prev_bev=None,
                use_prev_bev=1.0, can_bus=None, lidar2img=None, img_metas=None, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(prev_bev=prev_bev, use_prev_bev=use_prev_bev,
                                     can_bus=can_bus, lidar2img=lidar2img, img=img, **kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(
                img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img=None, prev_bev=None,
                     use_prev_bev=1.0, can_bus=None, lidar2img=None, **kwargs):
        # import pdb
        # pdb.set_trace()
        img = [img] if img is None else img
        bev_embed, outputs_classes, outputs_coords = self.simple_test(
            img, can_bus, lidar2img, prev_bev=prev_bev, use_prev_bev=use_prev_bev, **kwargs)
        self.prev_frame_info['prev_bev'] = bev_embed
        return bev_embed, outputs_classes, outputs_coords

    def simple_test_pts(self, x, can_bus=None, lidar2img=None,
                        prev_bev=None, use_prev_bev=1.0, image_shape=None, rescale=False):
        """Test function"""
        # image_metas = []
        # for i in range(len(img_metas)):
        #    lidar2img = [torch.from_numpy(l) for l in img_metas[i]['lidar2img']]
        #    #img_shape = [torch.from_numpy(s) for s in kwargs['img_metas'][i]['img_shape']]
        #    image_metas.append({'lidar2img': lidar2img, 'img_shape': torch.tensor(img_metas[i]['img_shape']), 'can_bus': torch.from_numpy(img_metas[i]['can_bus'])})
        outs = self.pts_bbox_head(
            x,
            can_bus,
            lidar2img,
            image_shape=image_shape,
            prev_bev=prev_bev,
            use_prev_bev=use_prev_bev)
        # import pdb
        # pdb.set_trace()
        # print('done')
        # torch.onnx.export(self.pts_bbox_head, (x, image_metas, prev_bev), 'pts_bbox_head.onnx', verbose=True, opset_version=16, dynamic_axes=None)
        # pdb.set_trace()
        # outs = self.pts_bbox_head(x, image_metas, prev_bev=prev_bev)
        # torch.onnx.export(self.pts_bbox_head, (x, image_metas, prev_bev), 'pts_bbox_head.onnx', verbose=True, opset_version=16, dynamic_axes=None)
        # pdb.set_trace()

        outs = {
            'bev_embed': outs[0],
            'all_cls_scores': outs[1],
            'all_bbox_preds': outs[2],
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        # return outs, outs
        return outs

    def simple_test(self, img=None, can_bus=None, lidar2img=None,
                    prev_bev=None, use_prev_bev=1.0, rescale=False):
        """Test function without augmentaiton."""
        from datetime import datetime
        t = datetime.now()
        # import pdb
        # pdb.set_trace()
        img_feats = self.extract_feat(img=img)
        print(datetime.now() - t)

        t = datetime.now()
        image_shape = img.shape[-2:]
        outs = self.simple_test_pts(
            img_feats, can_bus, lidar2img, prev_bev, use_prev_bev=use_prev_bev, image_shape=image_shape, rescale=rescale)
        print(datetime.now() - t)
        return outs['bev_embed'], outs['all_cls_scores'], outs['all_bbox_preds']

    def get_bboxes(self, outs, img_metas, rescale=False):
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def post_process(self, outputs_classes, outputs_coords):
        dic = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords}
        result_list = self.pts_bbox_head.get_bboxes(dic)
        return result_list
