import sys
import threading
import multiprocessing
import numpy as np
import sounddevice as sd
import cv2
import time
import math


class AudioBuffer:
    def __init__(self,):
        self.audio_sample_rate = 16000
        self.audio_packet_size = 320
        self.audio_sample_idx = 0       # index of the first sample in packet buffer
        self.audio_packet_buffer = []   # list of ndarrays
        self.audio_buffer_lock = multiprocessing.Lock()
        
        self.audio_buffering_time = 1.0 # in seconds, negative means infinite buffering


    def add_audio(self, audio: np.ndarray):
        audio_buffer = None

        # check whether the last audio packet is full
        with self.audio_buffer_lock:
            if len(self.audio_packet_buffer) > 0:
                if len(self.audio_packet_buffer[-1]) < self.audio_packet_size:
                    audio_buffer = self.audio_packet_buffer.pop(-1)

        # append audio_buffer before audio
        if audio_buffer is not None:
            audio = np.concatenate((audio_buffer, audio))

        # split audio into packets
        packets = []
        full_packets = len(audio) // self.audio_packet_size
        for i in range(full_packets):
            packets.append(audio[i*self.audio_packet_size : (i+1)*self.audio_packet_size])

        if full_packets * self.audio_packet_size < len(audio):
            packet_tail = audio[full_packets * self.audio_packet_size:]
            packets.append(packet_tail)
        
        # add packets to buffer
        if len(packets) > 0:
            with self.audio_buffer_lock:
                self.audio_packet_buffer.extend(packets)

        # remove old packets if buffer is too long
        if self.audio_buffering_time > 0.0:
            # calculate the number of packets to be removed
            max_buffer_size = math.ceil(self.audio_buffering_time * self.audio_sample_rate / self.audio_packet_size)
            with self.audio_buffer_lock:
                if len(self.audio_packet_buffer) > max_buffer_size:
                    # remove old packets                
                    self.audio_packet_buffer = self.audio_packet_buffer[-max_buffer_size:]
                    self.audio_sample_idx += max_buffer_size * self.audio_packet_size


    def get_audio_packet(self, padding=True):
        if padding:
            with self.audio_buffer_lock:                
                if len(self.audio_packet_buffer) > 0:
                    packet = self.audio_packet_buffer.pop(0)
                else:
                    packet = np.zeros(self.audio_packet_size).astype(np.int16)

                # pad zeros if packet is not full
                if len(packet) < self.audio_packet_size:
                    packet_tail = np.zeros(self.audio_packet_size - len(packet)).astype(np.int16)
                    packet = np.concatenate((packet, packet_tail))

                self.audio_sample_idx += self.audio_packet_size
                return packet

        else:
            with self.audio_buffer_lock:
                if len(self.audio_packet_buffer) > 0 and len(self.audio_packet_buffer[0]) == self.audio_packet_size:
                    self.audio_sample_idx += self.audio_packet_size
                    return self.audio_packet_buffer.pop(0)
                else:
                    return None


class VideoBuffer:
    def __init__(self,):
        self.video_frame_rate = 25.0
        self.video_frame_shape = (480, 640, 3)  # height, width, 3 bytes
        self.video_frame_idx = 0
        self.video_buffer = []
        self.video_buffer_lock = multiprocessing.Lock()

        self.video_buffering_time = 1.0 # in seconds, negative means infinite buffering


    def add_video_frame(self, video_frame: np.ndarray):
        # calculate max number of frames to be buffered
        max_buffer_size = math.ceil(self.video_buffering_time * self.video_frame_rate)

        with self.video_buffer_lock:
            if self.video_frame_shape != video_frame.shape:
                # update video shape
                self.video_frame_shape = video_frame.shape
            
            # remove old frames if buffer is too long
            if self.video_buffering_time > 0.0 and len(self.video_buffer) >= max_buffer_size:
                # set video_buffer length to max_buffer_size - 1
                self.video_buffer = self.video_buffer[-max_buffer_size+1:]

            self.video_buffer.append(video_frame)


    def get_video_frame(self, padding=True):
        if padding:
            with self.video_buffer_lock:
                if len(self.video_buffer) > 0:
                    dummy = False
                    video_frame = self.video_buffer.pop(0)
                else:
                    dummy = True
                    video_frame = np.zeros(self.video_frame_shape).astype(np.uint8)
    
            # print info if dummy frame
            if dummy:
                # print text on the frame
                text = "No video frame, frame_idx: {}".format(self.video_frame_idx)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (self.video_frame_shape[1] - text_size[0]) // 2
                text_y = (self.video_frame_shape[0] + text_size[1]) // 2
                cv2.putText(video_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)                
                
            self.video_frame_idx += 1
            return video_frame

        else:
            with self.video_buffer_lock:
                if len(self.video_buffer) > 0:
                    self.video_frame_idx += 1
                    return self.video_buffer.pop(0)
                else:
                    return None


class AVBuffer (AudioBuffer, VideoBuffer):
    def __init__(self,):
        AudioBuffer.__init__(self)
        VideoBuffer.__init__(self)


    def get_audio_video(self):
        # get an audio packet
        audio_packet = self.get_audio_packet(padding=True)

        # get start sample index of the next audio packet
        next_audio_sample_idx = self.audio_sample_idx
        
        # get next video frame matching to audio sample
        next_video_frame_sample_idx = int(round(self.video_frame_idx * self.audio_sample_rate / self.video_frame_rate))

        if next_video_frame_sample_idx > next_audio_sample_idx:
            video_frame = None
        else:
            video_frame = self.get_video_frame(padding=True)

        return audio_packet, video_frame


class AudioBufferPlayer:
    def __init__(self, audio_buffer: AudioBuffer):
        self.audio_buffer = audio_buffer


    def play(self, async_play=True):
        if async_play:
            # play audio in a separate thread
            play_thread = threading.Thread(target=self.play_async_thread)
            play_thread.start()
        else:
            # concatenate all audio packets
            audio_data = []
            while True:
                packet = self.audio_buffer.get_audio_packet(padding=False)
                if packet is None:
                    break
                audio_data.extend(packet)

            audio = np.array(audio_data).astype(np.int16)
            
            # play audio
            sd.play(audio, self.audio_buffer.audio_sample_rate)
            sd.wait()


    def play_async_thread(self):
        with sd.OutputStream(channels=1, 
                             callback=self.callback, 
                             samplerate=self.audio_buffer.audio_sample_rate, 
                             blocksize=self.audio_buffer.audio_packet_size):
            while True:
                pass


    def callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)

        audio_clip = self.audio_buffer.get_audio_packet(padding=True)
        audio_clip = audio_clip / 32768.0
        audio_clip = audio_clip.reshape(-1, 1)

        outdata[:] = audio_clip


class VideoBufferPlayer:
    def __init__(self, video_buffer: VideoBuffer, player_window_name='VideoBufferPlayer'):
        self.video_buffer = video_buffer

        self.player_window_name = player_window_name


    def play(self, async_play=True):
        if async_play:
            # play audio in a separate thread
            play_thread = threading.Thread(target=self.play_async_thread)
            play_thread.start()
        else:
            # get current time
            start_time = time.time()
            while True:
                frame = self.video_buffer.get_video_frame(padding=False)
                if frame is None:
                    break

                # calculate time to wait
                next_frame_time = start_time + self.video_buffer.video_frame_idx / self.video_buffer.video_frame_rate
                wait_time = int((next_frame_time - time.time()) * 1000)
                if wait_time <= 0:
                    wait_time = 1

                cv2.imshow(self.player_window_name, frame)
                cv2.waitKey(wait_time)


    def play_async_thread(self):
        # get current time
        start_time = time.time()

        while True:
            frame = self.video_buffer.get_video_frame(padding=True)
            if frame is None:
                break

            cv2.imshow(self.player_window_name, frame)            
            cv2.waitKey(1)

            # calculate time to wait
            next_frame_time = start_time + self.video_buffer.video_frame_idx / self.video_buffer.video_frame_rate
            wait_time = next_frame_time - time.time()
            if wait_time > 0.001:
                time.sleep(wait_time)


class AVBufferPlayer:
    def __init__(self, av_buffer: AVBuffer, player_window_name='AVBufferPlayer'):
        self.av_buffer = av_buffer

        self.player_window_name = player_window_name

        self.next_video_frame = None
        self.next_video_frame_lock = threading.Lock()


    def play(self):
        with sd.OutputStream(channels=1, 
                             callback=self.audio_callback,
                             samplerate=self.av_buffer.audio_sample_rate, 
                             blocksize=self.av_buffer.audio_packet_size):
            while True:
                with self.next_video_frame_lock:
                    frame = self.next_video_frame

                if frame is None:
                    # sleep for 1 ms
                    time.sleep(0.001)
                    continue

                cv2.imshow(self.player_window_name, frame)
                cv2.waitKey(1)


    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)

        audio_packet, video_frame = self.av_buffer.get_audio_video()

        audio_packet = audio_packet / 32768.0
        audio_packet = audio_packet.reshape(-1, 1)

        outdata[:] = audio_packet

        # update next video frame        
        if video_frame is not None:
            with self.next_video_frame_lock:
                self.next_video_frame = video_frame


def audio_test():
    sample_rate, audio_data = wavfile.read(r'C:\Users\yizzhan\Documents\Avatar\Audio2Head_Live\demo\audio\intro.wav')

    print('sample_rate:', sample_rate)
    print('audio_data.shape:', audio_data.shape)

    # plt.plot(audio_data)
    # plt.show()

    audio_buffer = AudioBuffer()
    audio_buffer_player = AudioBufferPlayer(audio_buffer)
    audio_buffer_player.play(async_play=True)

    # add audio
    add_step = 16000
    for i in range(0, audio_data.shape[0], add_step):
        audio_buffer.add_audio(audio_data[i:i+add_step])

        # calculate length of the added audio
        audio_length = min(add_step, audio_data.shape[0]-i)
        packet_num = len(audio_buffer.audio_packet_buffer)
        print('audio_length:', audio_length/sample_rate, ' packet_num:', packet_num)

        # sleep 1.5s
        time.sleep(0.6)


def video_test():
    # open camera
    cap = cv2.VideoCapture(0)

    # get video resolution
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_resolution = (video_height, video_width)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('video_resolution:', video_resolution)

    # create video buffer
    video_buffer = VideoBuffer()
    video_buffer.video_frame_rate = video_fps
    video_buffer_player = VideoBufferPlayer(video_buffer)
    video_buffer_player.play(async_play=True)

    # play video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_buffer.add_video_frame(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release camera
    cap.release()
    cv2.destroyAllWindows()


def audio_video_test():
    sample_rate, audio_data = wavfile.read(r'C:\Users\yizzhan\Documents\Avatar\Audio2Head_Live\demo\audio\intro.wav')

    print('sample_rate:', sample_rate)
    print('audio_data.shape:', audio_data.shape)

    # open camera
    cap = cv2.VideoCapture(0)

    # get video resolution
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_resolution = (video_height, video_width)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('video_resolution:', video_resolution)

    av_buffer = AVBuffer()
    av_buffer_player = AVBufferPlayer(av_buffer)

    # add audio
    av_buffer.add_audio(audio_data)

    av_buffer_player.play()
    


if __name__ == '__main__':
    from scipy.io import wavfile
    import matplotlib.pyplot as plt
    import time

    # audio_test()

    # video_test()

    audio_video_test()

