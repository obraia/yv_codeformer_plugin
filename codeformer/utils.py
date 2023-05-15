import cv2
import torch
from torchvision.transforms.functional import normalize
from codeformer.basicsr.utils import img2tensor, tensor2img
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper

class CodeFormer:
    def __init__(
            self, 
            model_path='weights/CodeFormer/codeformer.pth', 
            w=0.5, 
            upscale=2, 
            has_aligned=False, 
            only_center_face=False, 
            draw_box=False, 
            bg_upsampler='realesrgan',  
            bg_upsampler_model_path='weights/realesrgan/RealESRGAN_x2plus.pth',
            bg_tile=400
        ):
        self.w = w
        self.upscale = upscale
        self.has_aligned = has_aligned
        self.only_center_face = only_center_face
        self.draw_box = draw_box
        self.bg_upsampler = bg_upsampler
        self.bg_tile = bg_tile
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ------------------ set up background upsampler ------------------
        if self.bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                              'If you really want to use it, please modify the corresponding codes.',
                              category=RuntimeWarning)
                self.bg_upsampler = None
            else:
                from codeformer.basicsr.archs.rrdbnet_arch import RRDBNet
                from codeformer.basicsr.utils.realesrgan_utils import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                self.bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path=bg_upsampler_model_path,
                    model=model,
                    tile=self.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            self.bg_upsampler = None

        # ------------------ set up CodeFormer restorer -------------------
        self.net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                   connect_list=['32', '64', '128', '256']).to(self.device)

        ckpt_path = model_path
        checkpoint = torch.load(ckpt_path)['params_ema']
        self.net.load_state_dict(checkpoint)
        self.net.eval()

        # ------------------ set up FaceRestoreHelper -------------------
        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        self.face_helper = FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model = 'YOLOv5l',
            save_ext='png',
            use_parse=True,
            device=self.device)

    def restore(self, img):
        self.face_helper.clean_all()

        if self.has_aligned: 
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=self.only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=self.w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        # paste_back
        if not self.has_aligned:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.draw_box)
            
        if not self.has_aligned and restored_img is not None:
            return restored_img

       