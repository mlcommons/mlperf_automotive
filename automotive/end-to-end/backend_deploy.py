import torch
from mmcv import Config
from third_party.uniad_mmdet3d.models.builder import build_model
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel, DataContainer

# Add this import! It triggers the @MODELS.register_module() 
# decorators inside the UniAD codebase so MMCV knows what "UniAD" is.
import projects.mmdet3d_plugin.uniad

class BackendUniAD:
    def __init__(self, config_path, checkpoint_path, device_id=0):
        """
        Initializes the UniAD model based on OpenMMLab libraries.
        """
        self.device_id = device_id
        self.device = f'cuda:{device_id}'
        
        # Load the MMCV configuration
        cfg = Config.fromfile(config_path)
        
        # Build the model using the provided user snippets
        cfg.model.train_cfg = None
        self.model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
            
        # Load the checkpoint
        checkpoint = load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        
        # Send model to device and set to eval mode
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # We intentionally DO NOT wrap in MMDataParallel. 
        # Pickled DataContainers from the offline pipeline get deeply nested 
        # and confuse MMDataParallel's scatter function, causing subscript errors.

    def predict(self, batch_data):
        """
        Runs inference on a single batch of data.
        """
        # 1. Robustly unwrap DataContainers and move everything to the GPU.
        # Checking `type(obj).__name__` guarantees we catch pickled containers 
        # even if Python's strict `isinstance` fails due to import path mismatches.
        def unwrap_and_move(obj):
            if type(obj).__name__ == 'DataContainer' or isinstance(obj, DataContainer):
                return unwrap_and_move(obj.data)
            elif isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            elif isinstance(obj, list):
                return [unwrap_and_move(x) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(unwrap_and_move(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: unwrap_and_move(v) for k, v in obj.items()}
            return obj

        batch_data = unwrap_and_move(batch_data)
        
        # 2. Ensure 'img' is strictly formatted as a 5D batched tensor: [B, N, C, H, W]
        if 'img' in batch_data:
            img_data = batch_data['img']
            if isinstance(img_data, list) and len(img_data) > 0:
                if isinstance(img_data[0], torch.Tensor):
                    batch_data['img'] = torch.stack(img_data, dim=0)
                elif isinstance(img_data[0], list) and isinstance(img_data[0][0], torch.Tensor):
                    batch_data['img'] = torch.stack(img_data[0], dim=0).unsqueeze(0)
            
            if isinstance(batch_data['img'], torch.Tensor) and batch_data['img'].dim() == 4:
                batch_data['img'] = batch_data['img'].unsqueeze(0)

        # 3. Ensure 'img_metas' is strictly formatted for UniAD's forward_test.
        # UniAD hardcodes `metas = img_metas[0].data` in forward_test.
        
        if 'img_metas' in batch_data:
            metas = batch_data['img_metas']
            
            # Flatten to find the actual metadata dictionaries
            flat_metas = []
            def get_dicts(obj):
                if isinstance(obj, dict):
                    flat_metas.append(obj)
                elif isinstance(obj, (list, tuple)):
                    for i in obj: get_dicts(i)
                    
            get_dicts(metas)
            
            # Re-wrap using MMCV's DataContainer to satisfy the `.data` attribute check
            batch_data['img_metas'] = [DataContainer([flat_metas])]
        
        with torch.no_grad():
            outputs = self.model(return_loss=False, rescale=True, **batch_data)
            
        return outputs