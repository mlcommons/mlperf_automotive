import torch
import backend

import utils
import network
import torch


class BackendDeploy(backend.Backend):
    def __init__(self, model_path, num_classes, output_stride):
        super(BackendDeploy, self).__init__()
        self.checkpoint_file = model_path
        self.num_classes = num_classes
        self.output_stride = output_stride

    def version(self):
        return torch.__version__

    def name(self):
        return "python-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        model = network.modeling.__dict__['deeplabv3plus_resnet50'](
            num_classes=self.num_classes, output_stride=self.output_stride)
        checkpoint = torch.load(
            self.checkpoint_file, map_location=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.to(device=self.device)
        model.eval()
        self.model = model
        return self

    def predict(self, input):
        outputs = self.model(torch.from_numpy(input).to(device=self.device))
        return outputs.detach()
