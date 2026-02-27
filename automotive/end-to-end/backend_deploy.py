import torch
from mmcv import Config
from third_party.uniad_mmdet3d.models.builder import build_model
from mmcv.runner import load_checkpoint, wrap_fp16_model

# Add this import! It triggers the @MODELS.register_module()
import projects.mmdet3d_plugin.uniad

class BackendUniAD:
    def __init__(self, config_path, checkpoint_path, device_id=0):
        """
        Initializes the UniAD model based on OpenMMLab libraries.
        """
        self.device_id = device_id
        self.device = f'cuda:{device_id}'
        
        cfg = Config.fromfile(config_path)
        
        cfg.model.train_cfg = None
        self.model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)

        checkpoint = load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def predict(self, batch_data):
        """
        Runs inference on a single batch of data.
        """
        def move_to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            elif isinstance(obj, list):
                return [move_to_device(x) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            return obj

        batch_data = move_to_device(batch_data)

        batch_data['img_metas'] = [[batch_data['img_metas']]]

        with torch.no_grad():
            outputs = self.model(return_loss=False, rescale=True, **batch_data)
            
        # Reduce output size by only keeping the 'planning' data required for the accuracy checker
        filtered_outputs = []
        for out in outputs:
            filtered_outputs.append({'planning': out['planning']})
        return filtered_outputs