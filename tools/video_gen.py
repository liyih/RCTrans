import cv2
import os
from tqdm import tqdm
def merge_image_to_video(folder_name):
    fps = 25
    firstflag = True
    for f1 in tqdm(os.listdir(folder_name)[:30]):
        filename = os.path.join(folder_name, f1)
        frame = cv2.imread(filename)
        if firstflag == True:  # 读取第一张图时进行初始化，尺寸也近照些图
            firstflag = False
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            img_size = (frame.shape[1], frame.shape[0])
            video = cv2.VideoWriter("output.mp4", fourcc, fps, img_size)
        for index in range(fps):
            frame = cv2.imread(filename)
            frame_suitable = cv2.resize(frame, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
            video.write(frame_suitable)
    video.release()


if __name__ == '__main__':
    folder_name = 'new_pic'
    merge_image_to_video(folder_name)
