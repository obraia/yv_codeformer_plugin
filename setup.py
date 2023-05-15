import os
import sys
import subprocess

plugin_dir = os.path.dirname(__file__)
codeformer_dir = os.path.join(plugin_dir, 'codeformer')
weights_dir = os.path.join(codeformer_dir, 'weights')

def setup():
    download_weights()
    install_requirements()
    append_python_paths()

def download_weights():
    weights = [
            {
                'path': os.path.join(weights_dir, 'CodeFormer', 'codeformer.pth'),
                'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
            },
            {
                'path': os.path.join(weights_dir, 'facelib', 'detection_Resnet50_Final.pth'),
                'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
            },
            {
                'path': os.path.join(weights_dir, 'facelib', 'parsing_parsenet.pth'),
                'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
            },
            {
                'path': os.path.join(weights_dir, 'realesrgan', 'RealESRGAN_x2plus.pth'),
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            },
 
        ]

    for weight in weights:
        weight_path = weight['path']
        weight_url = weight['url']

        if not os.path.exists(weight_path):
            os.system(f'wget {weight_url} -O {weight_path}')

def install_requirements():
    requirements_path = os.path.join(plugin_dir, 'requirements.txt')
    out = subprocess.check_output(['pip', 'install', '-r', requirements_path])

    for line in out.splitlines():
        print(line)
        
def append_python_paths():
    if plugin_dir not in sys.path:
        sys.path.append(plugin_dir)

    if codeformer_dir not in sys.path:
        sys.path.append(codeformer_dir)
