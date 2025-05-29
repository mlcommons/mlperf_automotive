import torch
import backend

from cognata import Cognata, prepare_cognata
from transform import SSDTransformer
import importlib
from utils import generate_dboxes, Encoder
import cognata_labels
from model import SSD, ResNet
import numpy as np


class BackendDeploy(backend.Backend):
    def __init__(self, config, data_path,
                 checkpoint, nms_threshold, device='cpu'):
        super(BackendDeploy, self).__init__()
        self.config = importlib.import_module('config.' + config)
        self.image_size = self.config.model['image_size']
        self.og_image_size = self.config.model['og_image_size']
        self.device = device
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
        return "debug-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        model = SSD(
            self.config.model,
            backbone=ResNet(
                self.config.model),
            num_classes=self.num_classes)
        checkpoint = torch.load(self.checkpoint, map_location=self.device)
        self.og_image_size = self.config.model['og_image_size']
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model
        return self

    def predict(self, input):
        with torch.no_grad():
            model_input = input[0]
            img = torch.from_numpy(model_input).to(self.device)
            ploc, plabel = self.model(img)
            ploc, plabel = ploc.float(), plabel.float()
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
        return np.stack(results)
