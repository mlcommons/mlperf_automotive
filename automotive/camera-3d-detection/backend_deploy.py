import torch
import backend

import onnxruntime as ort
import numpy as np
from post_process import PostProcess


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class BackendDeploy(backend.Backend):
    def __init__(self, cfg, checkpoint):
        super(BackendDeploy, self).__init__()
        self.prev_frame_info = {
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.prev_bev = np.zeros(shape=(cfg.bev_h_ * cfg.bev_w_, 1, cfg._dim_), dtype=np.float32)
        self.checkpoint = checkpoint
        self.post_process = PostProcess(
            num_classes=10, max_num=300, pc_range=[
                -51.2, -51.2, -5.0, 51.2, 51.2, 3.0], post_center_range=[
                -61.2, -61.2, -10.0, 61.2, 61.2, 10.0], score_threshold=None)

    def version(self):
        return torch.__version__

    def name(self):
        return "onnx-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        self.ort_sess = ort.InferenceSession(self.checkpoint)
        self.input_img_name = self.ort_sess.get_inputs()[0].name
        self.input_prev_bev_name = self.ort_sess.get_inputs()[1].name
        self.input_use_prev_bev_name = self.ort_sess.get_inputs()[2].name
        self.input_can_bus_name = self.ort_sess.get_inputs()[3].name
        self.input_lidar2img_name = self.ort_sess.get_inputs()[4].name
        return self

    def predict(self, input):
        input_dict = input[0]
        input_data = {self.input_img_name: input_dict['img'],
                      self.input_prev_bev_name: self.prev_bev,
                      self.input_use_prev_bev_name: input_dict['use_prev_bev'],
                      self.input_can_bus_name: input_dict['can_bus'],
                      self.input_lidar2img_name: input_dict['lidar2img'],
                      }

        result = self.ort_sess.run(None, input_data)
        self.prev_bev = result[0]
        outputs_classes = torch.from_numpy(result[1])
        outputs_coords = torch.from_numpy(result[2])
        return outputs_classes, outputs_coords
