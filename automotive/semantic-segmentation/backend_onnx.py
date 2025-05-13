import backend
import torch
import onnxruntime as ort
import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class BackendOnnx(backend.Backend):
    def __init__(self, model_path, num_classes, output_stride):
        super(BackendOnnx, self).__init__()
        self.checkpoint = model_path
        self.num_classes = num_classes
        self.output_stride = output_stride

    def version(self):
        return torch.__version__

    def name(self):
        return "onnx-SUT"

    def load(self):
        self.ort_sess = ort.InferenceSession(self.checkpoint)
        self.input_img_name = self.ort_sess.get_inputs()[0].name
        return self

    def predict(self, inputs):
        model_input = inputs
        input_data = {self.input_img_name: to_numpy(model_input)}
        output = self.ort_sess.run(None, input_data)
        preds = torch.from_numpy(output[0])
        return preds
