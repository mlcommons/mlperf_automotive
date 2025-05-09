import torch
import backend

from cognata import Cognata, prepare_cognata
from transform import SSDTransformer
import importlib
from utils import generate_dboxes, Encoder
import cognata_labels
from model import SSD, ResNet


class BackendDeploy(backend.Backend):
    def __init__(self, config, data_path,
                 checkpoint, nms_threshold):
        super(BackendDeploy, self).__init__()
        self.config = importlib.import_module('config.' + config)
        self.image_size = self.config.model['image_size']
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SSD(
            self.config.model,
            backbone=ResNet(
                self.config.model),
            num_classes=self.num_classes)
        checkpoint = torch.load(self.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        self.model = model
        return self

    def predict(self, input):
        with torch.no_grad():
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            model_input = input[0]
            img = model_input[0].to(device).unsqueeze(0)
            img_id = model_input[1]
            img_size = model_input[2]
            ploc, plabel = self.model(img)
            ploc, plabel = ploc.float(), plabel.float()
            dts = []
            labels = []
            scores = []
            ids = []
            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                result = self.encoder.decode_batch(
                    ploc_i, plabel_i, self.nms_threshold, 500)[0]
                height, width = img_size
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    dts.append([loc_[0] * width, loc_[1] * height,
                               loc_[2] * width, loc_[3] * height,])
                    labels.append(label_)
                    scores.append(prob_)
                    ids.append(img_id)
        return dts, labels, scores, ids
