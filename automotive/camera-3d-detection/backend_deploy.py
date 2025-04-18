import torch
import backend

import onnxruntime as ort

from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
import numpy as np


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class BackendDeploy(backend.Backend):
    def __init__(self, cfg, checkpoint):
        super(BackendDeploy, self).__init__()
        self.model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        self.prev_frame_info = {
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.prev_bev = torch.zeros(cfg.bev_h_ * cfg.bev_w_, 1, cfg._dim_)
        self.checkpoint = checkpoint

    def version(self):
        return torch.__version__

    def name(self):
        return "onnx-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        self.model = MMDataParallel(self.model, device_ids=[0])
        self.model.eval()
        self.ort_sess = ort.InferenceSession(self.checkpoint)
        self.input_img_name = self.ort_sess.get_inputs()[0].name
        self.input_prev_bev_name = self.ort_sess.get_inputs()[1].name
        self.input_use_prev_bev_name = self.ort_sess.get_inputs()[2].name
        self.input_can_bus_name = self.ort_sess.get_inputs()[3].name
        self.input_lidar2img_name = self.ort_sess.get_inputs()[4].name
        return self

    def predict(self, input):
        input_dict = input[0]
        tmp_pos = (input_dict['can_bus'][0][:3]).copy()
        tmp_angle = (input_dict['can_bus'][0][-1]).copy()
        if input_dict["scene_token"] != self.prev_frame_info["scene_token"]:
            use_prev_bev = torch.tensor(0.0)
            # prev_bev = None
            input_dict["can_bus"][0][-1] = 0
            input_dict["can_bus"][0][:3] = 0
        else:
            use_prev_bev = torch.tensor(1.0)
            input_dict["can_bus"][0][:3] -= self.prev_frame_info["prev_pos"]
            input_dict["can_bus"][0][-1] -= self.prev_frame_info["prev_angle"]
        self.prev_frame_info["scene_token"] = input_dict["scene_token"]
        can_bus = input_dict["can_bus"][0].astype(np.float32)
        lidar2img = np.stack(
            input_dict['lidar2img'][0]).astype(
            np.float32)
        img = input_dict['img'][0]
        input_data = {self.input_img_name: np.expand_dims(to_numpy(img), 0),
                      self.input_prev_bev_name: to_numpy(self.prev_bev),
                      self.input_use_prev_bev_name: to_numpy(use_prev_bev),
                      self.input_can_bus_name: can_bus,
                      self.input_lidar2img_name: np.expand_dims(lidar2img, 0),
                      }

        result = self.ort_sess.run(None, input_data)
        bev_embed = torch.from_numpy(result[0])
        outputs_classes = torch.from_numpy(result[1])
        outputs_coords = torch.from_numpy(result[2])
        result = self.model.module.post_process(
            outputs_classes, outputs_coords)
        self.prev_bev = bev_embed
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        return result
