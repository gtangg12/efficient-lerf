from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.utils.visualization import visualize_image


class ModelInstructPix2Pix:
    """
    """
    def __init__(self, method='timbrooks/instruct-pix2pix', device='cuda'):
        """
        """
        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            method, torch_dtype=torch.float32, safety_checker=None
        )
        self.model.to(device)
        self.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.model.scheduler.config
        )

    def __call__(
        self,
        text: str | list[str],
        image: TorchTensor[..., 'H', 'W', 3], 
        steps = 50,
        downsample=1,
        **kwargs
    ) -> TorchTensor[..., 'H', 'W', 3]:
        """
        """
        H, W, _ = image.shape[-3:]

        if isinstance(text, str):
            text = [text]
        if image.ndim == 3:
            inputs = image.unsqueeze(0)
        else:
            inputs = image
            if len(text) == 1 and len(image) > 1:
                text = text * len(image)
        inputs = [
            visualize_image(x.numpy()).resize((W // downsample, H // downsample)) 
            for x in inputs
        ]
        with torch.no_grad():
            outputs = self.model(text, image=inputs, num_inference_steps=steps, **kwargs)[0]
        outputs_resized = [
            torch.from_numpy(np.array(x.resize((W, H)))) 
            for x in outputs
        ]
        return outputs_resized[0] if image.ndim == 3 else outputs_resized


if __name__ == '__main__':
    import os
    tests = Path('tests') / 'model_instructpix2pix'
    os.makedirs(tests, exist_ok=True)

    model = ModelInstructPix2Pix()

    image = Image.open(tests / 'camp.jpg')
    image = image.resize((image.width // 4, image.height // 4))
    image = torch.from_numpy(np.array(image))[None, ...].repeat(8, 1, 1, 1)
    print(image.shape)
    
    output = model('make it snowy', image)
    visualize_image(output[0].numpy()).save(tests / 'camp_edited.jpg')