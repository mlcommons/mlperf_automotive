import torch
from mmcv import Config
from third_party.uniad_mmdet3d.models.builder import build_model
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel, DataContainer

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
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
            
        # Load the checkpoint
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        
        # Send model to device, set to eval mode, and wrap with MMDataParallel
        model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        if self.device.type == 'cuda':
            self.model = MMDataParallel(model, device_ids=[self.device_id])
        else:
            self.model = MMDataParallel(model)

    def predict(self, batch_data):
        """
        Takes collated batch data and runs inference through UniAD.
        """
        # Fix for UniAD MMDataParallel scatter unwrapping issue:
        # MMDataParallel strips the outer DataContainer.
        # Since uniad_e2e expects img_metas[0] to be a DataContainer (via .data access),
        # we double-wrap it so the inner DataContainer survives the scatter.
        if 'img_metas' in batch_data and isinstance(batch_data['img_metas'], list):
            if len(batch_data['img_metas']) > 0 and isinstance(batch_data['img_metas'][0], DataContainer):
                orig_data = batch_data['img_metas'][0].data
                batch_data['img_metas'][0] = DataContainer([DataContainer(orig_data, cpu_only=True)], cpu_only=True)
        elif 'img_metas' in batch_data and isinstance(batch_data['img_metas'], DataContainer):
            orig_data = batch_data['img_metas'].data
            if isinstance(orig_data, list) and len(orig_data) > 0 and not isinstance(orig_data[0], DataContainer):
                batch_data['img_metas'] = DataContainer([DataContainer(d, cpu_only=True) for d in orig_data], cpu_only=True)

        # Fix for img tensor dimension issue:
        # extract_img_feat expects img to be a 5D tensor (B, N_cams, C, H, W).
        # We check the shape directly to avoid accessing missing MMCV attributes.
        if 'img' in batch_data:
            if isinstance(batch_data['img'], DataContainer):
                img_data = batch_data['img'].data
                if isinstance(img_data, list) and len(img_data) > 0 and isinstance(img_data[0], torch.Tensor):
                    batch_data['img'] = DataContainer(torch.stack(img_data, dim=0), stack=True)
                elif isinstance(img_data, torch.Tensor) and img_data.dim() == 4:
                    batch_data['img'] = DataContainer(img_data.unsqueeze(0), stack=True)
            elif isinstance(batch_data['img'], list) and len(batch_data['img']) > 0:
                dc = batch_data['img'][0]
                if isinstance(dc, DataContainer):
                    img_data = dc.data
                    if isinstance(img_data, list) and len(img_data) > 0 and isinstance(img_data[0], torch.Tensor):
                        batch_data['img'][0] = DataContainer(torch.stack(img_data, dim=0), stack=True)
                    elif isinstance(img_data, torch.Tensor) and img_data.dim() == 4:
                        batch_data['img'][0] = DataContainer(img_data.unsqueeze(0), stack=True)

        with torch.no_grad():
            # In mmcv, MMDataParallel automatically unpacks DataContainers 
            # return_loss=False switches to inference test_step forward
            outputs = self.model(return_loss=False, rescale=True, **batch_data)
            
        return outputs