import os
import cv2
from codeformer.utils import CodeFormer

images_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'images')
weights_dir = os.path.join(os.path.dirname(__file__), 'codeformer', 'weights')

def handler(data, properties):
    upscale = properties.get('upscale')
    weight = properties.get('weight')

    for item in data:
      image = cv2.imread(os.path.join(images_dir, item['image']))

      restorer = CodeFormer(
          model_path=os.path.join(weights_dir, 'CodeFormer', 'codeformer.pth'),
          w=float(weight),
          upscale=int(upscale),
          bg_upsampler='realesrgan',
          bg_upsampler_model_path=os.path.join(weights_dir, 'realesrgan', 'RealESRGAN_x2plus.pth'),
      )

      restored_image = restorer.restore(image)

      cv2.imwrite(os.path.join(images_dir, item['image']), restored_image)

    return data