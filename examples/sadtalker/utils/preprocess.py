import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image

# 3dmm extraction
import mindspore as ms
from models.face3d.utils import load_lm3d, align_img
from models.face3d.networks import define_net_recon

from scipy.io import savemat
from utils.croper import Preprocesser


import warnings

warnings.filterwarnings("ignore")


def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of ms.tensors

    Parameters:
        coeffs          -- ms.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }


class CropAndExtract():
    def __init__(self, config):

        self.propress = Preprocesser()
        self.net_recon = define_net_recon(
            net_recon='resnet50', use_last_fc=False, init_path='')

        checkpoint_dir = config.path.checkpoint_dir
        path_net_recon = os.path.join(checkpoint_dir, config.path.path_of_net_recon_model)
        path_bfm = os.path.join(checkpoint_dir, config.path.dir_of_bfm_fitting)

        param_dict = ms.load_checkpoint(path_net_recon)
        ms.load_param_into_net(self.net_recon, param_dict)
        self.net_recon.set_train(False)

        self.lm3d_std = load_lm3d(path_bfm)

    def generate(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):

        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]

        landmarks_path = os.path.join(save_dir, pic_name+'_landmarks.txt')
        coeff_path = os.path.join(save_dir, pic_name+'.mat')
        png_path = os.path.join(save_dir, pic_name+'.png')

        # load input
        if not os.path.isfile(input_path):
            raise ValueError(
                'input_path must be a valid path to video/image file')
        elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_frames = [cv2.imread(input_path)]
            fps = 25
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(input_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                full_frames.append(frame)
                if source_image_flag:
                    break

        x_full_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                         for frame in full_frames]

        print("start cropping the image ...")

        # crop images as the
        if 'crop' in crop_or_resize.lower():  # default crop
            x_full_frames, crop, quad = self.propress.crop(
                x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif 'full' in crop_or_resize.lower():
            x_full_frames, crop, quad = self.propress.crop(
                x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else:  # resize mode
            oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1]
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        frames_pil = [Image.fromarray(cv2.resize(
            frame, (pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None

        print("finished cropping, now saving the image to file.")

        # save crop info
        for frame in frames_pil:
            cv2.imwrite(png_path, cv2.cvtColor(
                np.array(frame), cv2.COLOR_RGB2BGR))

        print(f"finished cropping the image and saved to file {png_path}.")

        # 2. get the landmark according to the detected face.
        if not os.path.isfile(landmarks_path):
            lm = self.propress.predictor.extract_keypoint(
                frames_pil, landmarks_path)
        else:
            print(' Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch
            video_coeffs, full_coeffs = [],  []
            for idx in tqdm(range(len(frames_pil)), desc='3DMM Extraction In Video:'):
                frame = frames_pil[idx]
                W, H = frame.size
                lm1 = lm[idx].reshape([-1, 2])

                if np.mean(lm1) == -1:
                    lm1 = (self.lm3d_std[:, :2]+1)/2.
                    lm1 = np.concatenate(
                        [lm1[:, :1]*W, lm1[:, 1:2]*H], 1
                    )
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                trans_params, im1, lm1, _ = align_img(
                    frame, lm1, self.lm3d_std)

                trans_params = np.array(
                    [float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_t = ms.Tensor(np.array(
                    im1)/255., dtype=ms.float32).permute(2, 0, 1).unsqueeze(0)

                full_coeff = self.net_recon(im_t)
                coeffs = split_coeff(full_coeff)

                pred_coeff = {key: coeffs[key].asnumpy() for key in coeffs}

                pred_coeff = np.concatenate([
                    pred_coeff['exp'],
                    pred_coeff['angle'],
                    pred_coeff['trans'],
                    trans_params[2:][None],
                ], 1)
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.asnumpy())

            semantic_npy = np.array(video_coeffs)[:, 0]

            savemat(coeff_path, {'coeff_3dmm': semantic_npy,
                    'full_3dmm': np.array(full_coeffs)[0]})

        return coeff_path, png_path, crop_info
