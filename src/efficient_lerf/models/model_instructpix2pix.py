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
        text : str | list[str],
        image: TorchTensor[..., 'H', 'W', 3], 
        mask : TorchTensor[..., 'H', 'W'] = None, 
        steps = 50,
        **kwargs
    ) -> TorchTensor['H', 'W', 3]:
        """
        """
        if isinstance(text, str):
            text = [text]
        if image.ndim == 3:
            input = image.unsqueeze(0)
            if mask is not None: 
                mask = mask.unsqueeze(0)
        else:
            input = image
            if len(text) == 1 and len(image) > 1:
                text = text * len(image)
        input = [visualize_image(x.numpy()) for x in input]

        # TODO:: implement batching
        #input, text = input[:1], text[:1]

        with torch.no_grad():
            outputs = self.model(text, image=input, num_inference_steps=steps, **kwargs)[0]
        outputs_resized = [
            torch.from_numpy(np.array(x.resize(input[0].size))) 
            for x in outputs
        ]
        #visualize_image(outputs_resized[0].cpu().numpy()).save('000_raw.png')
        edited = []
        if mask is not None:
            for x, im, m in zip(outputs_resized, input, mask.unsqueeze(-1)):
                edited.append(m * x + ~m * torch.tensor(np.array(im)))
        edited = torch.stack(edited)
        if image.ndim == 3:
            edited = edited[0]
        return edited


if __name__ == '__main__':
    import os
    tests = Path('tests') / 'model_instructpix2pix'
    os.makedirs(tests, exist_ok=True)

    model = ModelInstructPix2Pix()

    image = Image.open(tests / 'camp.jpg')
    image = image.resize((image.width // 4, image.height // 4))
    image = torch.from_numpy(np.array(image))
    print(image.shape)
    
    output = model('make it snowy', image)
    visualize_image(output.numpy()).save(tests / 'camp_edited.jpg')