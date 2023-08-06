import argparse
import subprocess
import python_speech_features
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import pyworld
import torch
from modules.audio2pose import get_pose_from_audio, audio2poseLSTM, audio2poseLSTM_Live
from skimage import io, img_as_float32
import cv2
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.audio2kp import AudioModel3D
import yaml,os,imageio
import threading


def draw_annotation_box( image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

def inter_pitch(y,y_flag):
    frame_num = y.shape[0]
    i = 0
    last = -1
    while(i<frame_num):
        if y_flag[i] == 0:
            while True:
                if y_flag[i]==0:
                    if i == frame_num-1:
                        if last !=-1:
                            y[last+1:] = y[last]
                        i+=1
                        break
                    i+=1
                else:
                    break
            if i >= frame_num:
                break
            elif last == -1:
                y[:i] = y[i]
            else:
                inter_num = i-last+1
                fy = np.array([y[last],y[i]])
                fx = np.linspace(0, 1, num=2)
                f = interp1d(fx,fy)
                fx_new = np.linspace(0,1,inter_num)
                fy_new = f(fx_new)
                y[last+1:i] = fy_new[1:-1]
                last = i
                i+=1

        else:
            last = i
            i+=1
    return y

def get_audio_feature_from_audio(audio_path,norm = True):
    sample_rate, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]
    if norm:
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        a = python_speech_features.mfcc(audio, sample_rate)
        b = python_speech_features.logfbank(audio, sample_rate)
        c, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
        c_flag = (c == 0.0) ^ 1
        c = inter_pitch(c, c_flag)
        c = np.expand_dims(c, axis=1)
        c_flag = np.expand_dims(c_flag, axis=1)
        frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])

        c = c * 0
        c_flag = c_flag * 0

        cat = np.concatenate([a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1)
        return cat

def audio2head(audio_path, img_path, model_path, save_path):
    temp_audio="./results/temp.wav"
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, temp_audio))
    output = subprocess.call(command, shell=True, stdout=None)

    audio_feature = get_audio_feature_from_audio(temp_audio)
    frames = len(audio_feature) // 4

    img = io.imread(img_path)[:, :, :3]
    img = cv2.resize(img, (256, 256))

    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()


    ref_pose_rot, ref_pose_trans = get_pose_from_audio(img, audio_feature, model_path)
    torch.cuda.empty_cache()

    config_file = r"./config/vox-256.yaml"
    with open(config_file) as f:
        config = yaml.load(f)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()

    opt = argparse.Namespace(**yaml.load(open("./config/parameters.yaml")))
    audio2kp = AudioModel3D(opt).cuda()

    checkpoint  = torch.load(model_path)
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    generator.load_state_dict(checkpoint["generator"])
    audio2kp.load_state_dict(checkpoint["audio2kp"])

    generator.eval()
    kp_detector.eval()
    audio2kp.eval()
    
    audio_f = []
    poses = []
    pad = np.zeros((4,41),dtype=np.float32)
    for i in range(0, frames, opt.seq_len // 2):
        temp_audio = []
        temp_pos = []
        for j in range(opt.seq_len):
            if i + j < frames:
                temp_audio.append(audio_feature[(i+j)*4:(i+j)*4+4])
                trans = ref_pose_trans[i + j]
                rot = ref_pose_rot[i + j]
            else:
                temp_audio.append(pad)
                trans = ref_pose_trans[-1]
                rot = ref_pose_rot[-1]

            pose = np.zeros([256, 256])
            draw_annotation_box(pose, np.array(rot), np.array(trans))
            temp_pos.append(pose)
        audio_f.append(temp_audio)
        poses.append(temp_pos)
        
    audio_f = torch.from_numpy(np.array(audio_f,dtype=np.float32)).unsqueeze(0)
    poses = torch.from_numpy(np.array(poses, dtype=np.float32)).unsqueeze(0)

    bs = audio_f.shape[1]
    predictions_gen = []
    total_frames = 0

    import time

    start_time = time.time()    

    for bs_idx in range(bs):
        curr_time = time.time()
        print("processing batch %d/%d, time elapsed %f" % (bs_idx, bs, curr_time - start_time))
        start_time = curr_time

        t = {}

        t["audio"] = audio_f[:, bs_idx].cuda()
        t["pose"] = poses[:, bs_idx].cuda()
        t["id_img"] = img
        kp_gen_source = kp_detector(img)

        gen_kp = audio2kp(t)
        if bs_idx == 0:
            startid = 0
            end_id = opt.seq_len // 4 * 3
        else:
            startid = opt.seq_len // 4
            end_id = opt.seq_len // 4 * 3

        for frame_bs_idx in range(startid, end_id):
            tt = {}
            tt["value"] = gen_kp["value"][:, frame_bs_idx]
            if opt.estimate_jacobian:
                tt["jacobian"] = gen_kp["jacobian"][:, frame_bs_idx]
            out_gen = generator(img, kp_source=kp_gen_source, kp_driving=tt)
            out_gen["kp_source"] = kp_gen_source
            out_gen["kp_driving"] = tt
            del out_gen['sparse_deformed']
            del out_gen['occlusion_map']
            del out_gen['deformed']
            predictions_gen.append(
                (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))

            total_frames += 1
            if total_frames >= frames:
                break
        if total_frames >= frames:
            break

    log_dir = save_path
    if not os.path.exists(os.path.join(log_dir, "temp")):
        os.makedirs(os.path.join(log_dir, "temp"))
    image_name = os.path.basename(img_path)[:-4]+ "_" + os.path.basename(audio_path)[:-4] + ".mp4"

    video_path = os.path.join(log_dir, "temp", image_name)

    imageio.mimsave(video_path, predictions_gen, fps=25.0)

    save_video = os.path.join(log_dir, image_name)
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
    os.system(cmd)
    os.remove(video_path)


class TalkingHeadGenerator():
    # init function
    def __init__(self):
        self.audio_sample_rate = 16000
        self.in_audio = None
        self.generation_seq_len = 32    # should be even, max 64

        # packets pushed in append_audio()
        self.audio_packet = []
        self.audio_feature_packet = []
        self.pose_packet = []
        # packets pushed in synthesize_image_thread()
        self.annotation_img_packet = []
        self.portrait_img_packet = []
        self.packet_lock = threading.Lock()

        self.img = None                 # the source image
        self.kp_gen_source = None       # the source image's keypoints

        self.pose_generator = None
        self.kp_detector = None
        self.img_generator = None
        self.audio2kp = None

        self.pose_x = {}                # inference input for pose generator


    def set_img(self, img_path, model_path):
        # load image
        img = io.imread(img_path)[:, :, :3]
        img = cv2.resize(img, (256, 256))

        img = np.array(img_as_float32(img))
        img = img.transpose((2, 0, 1))
        self.img = torch.from_numpy(img).unsqueeze(0).cuda()

        self.pose_x["img"] = self.img

        # set image generator
        config_file = r"./config/vox-256.yaml"
        with open(config_file) as f:
            config = yaml.load(f)
        self.kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                      **config['model_params']['common_params'])
        self.img_generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        self.kp_detector = self.kp_detector.cuda()
        self.img_generator = self.img_generator.cuda()

        opt = argparse.Namespace(**yaml.load(open("./config/parameters.yaml")))
        self.audio2kp = AudioModel3D(opt).cuda()

        checkpoint  = torch.load(model_path)
        self.kp_detector.load_state_dict(checkpoint["kp_detector"])
        self.img_generator.load_state_dict(checkpoint["generator"])
        self.audio2kp.load_state_dict(checkpoint["audio2kp"])

        self.img_generator.eval()
        self.kp_detector.eval()
        self.audio2kp.eval()

        self.kp_gen_source = self.kp_detector(self.img)


    def set_pose_generator(self, model_path):
        self.pose_generator = audio2poseLSTM_Live().cuda()

        ckpt_para = torch.load(model_path)

        self.pose_generator.load_state_dict(ckpt_para["audio2pose"])
        self.pose_generator.eval()


    def append_audio(self, audio_clip):
        mfcc_winlen = 0.025
        mfcc_winstep = 0.01
        mfcc_winlen_sample = int(self.audio_sample_rate * mfcc_winlen)
        mfcc_winstep_sample = int(self.audio_sample_rate * mfcc_winstep)

        minv = np.array([-0.639, -0.501, -0.47, -102.6, -32.5, 184.6], dtype=np.float32)
        maxv = np.array([0.411, 0.547, 0.433, 159.1, 116.5, 376.5], dtype=np.float32)


        if self.in_audio is None:
            self.in_audio = audio_clip
        else:
            self.in_audio = np.concatenate((self.in_audio, audio_clip), axis=0)

        # at least 4 mfcc steps are needed to create a packet
        if self.in_audio.shape[0] <= mfcc_winlen_sample + mfcc_winstep_sample * 4:
            return

        try:
            # extract audio feature        
            audio_feature = self.get_audio_feature_from_audio(self.in_audio, self.audio_sample_rate)

            # every 4 mfcc steps are packed into a packet
            audio_feature_len = audio_feature.shape[0]
            packet_num = audio_feature_len // 4

            # buffer each packet
            audio_seq = []
            for i in range(packet_num):
                audio_seq.append(audio_feature[i * 4: (i + 1) * 4, :])

            # calculate pose
            audio = torch.from_numpy(np.array(audio_seq,dtype=np.float32)).unsqueeze(0).cuda()

            self.pose_x["audio"] = audio
            poses = self.pose_generator(self.pose_x)

            poses = poses.cpu().data.numpy()[0]

            poses = (poses+1)/2*(maxv-minv)+minv
            rot, trans =  poses[:,:3].copy(), poses[:,3:].copy()


            with self.packet_lock:
                for i in range(packet_num):
                    self.audio_packet.append(self.in_audio[i * mfcc_winstep_sample * 4: 
                                                        (i + 1) * mfcc_winstep_sample * 4])

                    self.audio_feature_packet.append(audio_feature[i * 4: (i + 1) * 4, :])

                    self.pose_packet.append((rot[i], trans[i]))


            # remove the used audio
            featured_audio_len = packet_num * mfcc_winstep_sample * 4
            self.in_audio = self.in_audio[featured_audio_len:]

        except:
            print("audio feature extraction error")

        # visualize audio feature
        # import matplotlib.pyplot as plt
        # plt.imshow(audio_feature.T, aspect='auto', origin='lower')
        # plt.show()


    def get_av_packet(self):
        pass


    def get_img(self, ref_pose_rot, ref_pose_trans):
        rot = ref_pose_rot[-1]
        trans = ref_pose_trans[-1]
        
        pose_ano_img = np.zeros([256, 256])
        draw_annotation_box(pose_ano_img, np.array(rot), np.array(trans))

        # show pose_ano_img using opencv
        cv2.imshow("pose_ano_img", pose_ano_img)
        cv2.waitKey(1)

        # # plot pose_ano_img
        # import matplotlib.pyplot as plt
        # plt.imshow(pose_ano_img)
        # plt.show()


    def get_audio_feature_from_audio(self, audio, sample_rate):
        audio = audio - np.mean(audio)
        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            audio = audio / audio_max

        a = python_speech_features.mfcc(audio, sample_rate)
        b = python_speech_features.logfbank(audio, sample_rate)

        # abort pitch feature, it's too slow
        c = np.zeros((a.shape[0], 1))
        c_flag = np.zeros((a.shape[0], 1))

        frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])
        cat = np.concatenate([a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1)

        return cat


    def generate_img_thread(self):
        pass


def main_old(parse):
    os.makedirs(parse.save_path,exist_ok=True)
    audio2head(parse.audio_path,parse.img_path,parse.model_path,parse.save_path)



if __name__ == '__main__':
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path",default=r"./demo/audio/intro.wav",help="audio file sampled as 16k hz")
    parser.add_argument("--img_path",default=r"./demo/img/paint.jpg", help="reference image")
    parser.add_argument("--save_path",default=r"./results", help="save path")
    parser.add_argument("--model_path",default=r"./checkpoints/audio2head.pth.tar", help="pretrained model path")

    parse = parser.parse_args()

    # main_old(parse)

    # read audio ./results/temp.wav using wavefile
    audio_path = parse.audio_path

    sample_rate, audio = wavfile.read(audio_path)

    # clip audio
    # audio = audio[0:16000*10]
    
    # print sample rate and audio shape
    print("sample rate: ", sample_rate)
    print("audio shape: ", audio.shape)

    # create TalkingHeadGenerator
    th_generator = TalkingHeadGenerator()
    th_generator.set_img(parse.img_path, parse.model_path)
    th_generator.set_pose_generator(parse.model_path)

    # for i in range(1, 1000):
    #     clip = audio[0:i]
    #     c = th_generator.get_audio_feature_from_audio(clip, sample_rate)

    #     # print i and c.shape
    #     print(clip.shape, c.shape)

    # exit(0)

    for i in tqdm(range(0, len(audio), 160)):
        # append audio
        th_generator.append_audio(audio[i:i+160])
        # # print info
        # print(i+160, 
        #       th_generator.in_audio.shape,
        #       len(th_generator.audio_packet),
        #       len(th_generator.audio_feature_packet),
        #       len(th_generator.pose_packet))

    # for each packet, generate image
    for i in tqdm(range(len(th_generator.audio_packet))):
        rot = th_generator.pose_packet[i][0]
        trans = th_generator.pose_packet[i][1]

        pose_ano_img = np.zeros([256, 256])
        draw_annotation_box(pose_ano_img, np.array(rot), np.array(trans))

        th_generator.annotation_img_packet.append(pose_ano_img)        

    # for each gen_seq, generate image
    for i in tqdm(range(0, len(th_generator.audio_packet), 32)):        
        if i+64 > len(th_generator.audio_packet):
            break

        audio_f = th_generator.audio_feature_packet[i:i+64]
        poses = th_generator.annotation_img_packet[i:i+64]
        audio_f = torch.from_numpy(np.array(audio_f,dtype=np.float32)).unsqueeze(0)
        poses = torch.from_numpy(np.array(poses,dtype=np.float32)).unsqueeze(0)

        t = {}
        t["audio"] = audio_f.cuda()
        t["pose"] = poses.cuda()
        t["id_img"] = th_generator.img

        gen_kp = th_generator.audio2kp(t)

        if i == 0:
            startid = 0
            end_id = 48
        else:
            startid = 16
            end_id = 48
        for frame_bs_idx in range(startid, end_id):
            tt = {}
            tt["value"] = gen_kp["value"][:, frame_bs_idx]
            tt["jacobian"] = gen_kp["jacobian"][:, frame_bs_idx]
            out_gen = th_generator.img_generator(th_generator.img, 
                                                 kp_source=th_generator.kp_gen_source, 
                                                 kp_driving=tt)

            out_img = (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8)
            th_generator.portrait_img_packet.append(out_img)


    # for i in range(len(th_generator.audio_packet)):
    #     pose_ano_img = th_generator.annotation_img_packet[i]

    #     # show pose_ano_img using opencv
    #     cv2.imshow("pose_ano_img", pose_ano_img)
    #     cv2.waitKey(1)


    for i in range(len(th_generator.portrait_img_packet)):
        out_img = th_generator.portrait_img_packet[i]

        # convert out_img to BGR
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        
        # show out_img using opencv
        cv2.imshow("out_img", out_img)
        cv2.waitKey(1)

    exit(0)

    # play the clipped audio
    import sounddevice as sd
    duration = len(th_generator.audio) / sample_rate
    sd.play(th_generator.audio, sample_rate)
    sd.wait(duration)

    # # plot audio
    # import matplotlib.pyplot as plt
    # plt.plot(audio)
    # plt.show()
