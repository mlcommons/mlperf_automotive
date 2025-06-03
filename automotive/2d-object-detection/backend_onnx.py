import backend
import torch
from cognata import prepare_cognata
import importlib
from utils import generate_dboxes, Encoder
import cognata_labels
import onnxruntime as ort
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class BackendOnnx(backend.Backend):
    def __init__(
            self, config, data_path,
            checkpoint, nms_threshold, device='cpu'):
        super(BackendOnnx, self).__init__()
        self.config = importlib.import_module('config.' + config)
        self.image_size = self.config.model['image_size']
        self.og_image_size = self.config.model['og_image_size']
        dboxes = generate_dboxes(self.config.model, model="ssd")
        folders = self.config.dataset['folders']
        cameras = self.config.dataset['cameras']
        if self.config.dataset['use_label_file']:
            label_map = cognata_labels.label_map
        else:
            _, label_map, _ = prepare_cognata(data_path, folders, cameras)
        self.num_classes = len(label_map.keys())
        self.checkpoint = checkpoint
        self.encoder = Encoder(dboxes)
        self.nms_threshold = nms_threshold

    def version(self):
        return torch.__version__

    def name(self):
        return "onnx-SUT"

    def load(self):
        self.ort_sess = ort.InferenceSession(self.checkpoint)
        self.input_img_name = self.ort_sess.get_inputs()[0].name
        return self

    def predict(self, inputs):
        with torch.inference_mode():
            model_input = inputs[0]
            input_data = {self.input_img_name: model_input}
            ploc, plabel = self.ort_sess.run(None, input_data)
            ploc, plabel = torch.from_numpy(
                ploc).float(), torch.from_numpy(plabel).float()
            results = []
            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                result = self.encoder.decode_batch(
                    ploc_i, plabel_i, self.nms_threshold, 500)[0]
                height, width = self.og_image_size
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    results.append([loc_[0] * width, loc_[1] * height,
                                    loc_[2] * width, loc_[3] * height, label_, prob_])
        if len(results) == 0:
            return results
        else:
            return np.stack(results)
