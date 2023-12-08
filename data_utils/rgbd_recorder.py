import argparse
import os
from os import makedirs
import shutil
import open3d as o3d
import time
import os
import glob
import cv2
import numpy as np
import tqdm
import imageio

def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        os.stat(path)
    except (OSError, ValueError):
        return False
    return True



def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        shutil.rmtree(path_folder)
        makedirs(path_folder)

def get_face_mask(image_path):
    import face_alignment
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    except:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    input_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
    input = cv2.cvtColor(input_raw, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(input)
    lands = preds[0].reshape(-1, 2)[:,:2]
    del fa
    x_min = np.min(lands[:,:1])
    x_max = np.max(lands[:,:1])
    y_min = np.min(lands[:,-1:])
    y_max = np.max(lands[:,-1:])
    x_len =int(1.8*(x_max - x_min)/10) *10
    y_len =int(1.8* (y_max - y_min)/10) *10
    f_len = x_len if x_len>y_len else y_len
    # print(x_len,y_len,f_len)
    x,y =  int(lands[28][0] - f_len/2) ,int(lands[28][1] - f_len/2)
    x = x if x>0 else 0
    y = y if y>0 else 0
    x_len = f_len if f_len < input.shape[1] - x else input.shape[1] - x
    y_len = f_len if f_len < input.shape[0] - y else input.shape[0] - y
    f_len = np.min([x_len,y_len])
    return x,y,f_len

def crop_face(path,name):
    raw_data_rgb_path = os.path.join(path,name,"raw","color")
    raw_data_depth_path = os.path.join(path,name,"raw","depth")
    rgb_files = glob.glob(os.path.join(raw_data_rgb_path, '*.jpg'))
    rgb_files = sorted(rgb_files)
    depth_files = glob.glob(os.path.join(raw_data_depth_path, '*.png'))
    depth_files = sorted(depth_files)
    x,y,f_len = get_face_mask(rgb_files[0])
    idx = 0
    count = 0
    rgbs = []
    depths = []
    for item in tqdm.tqdm(rgb_files):
        if(count%6 != 0 ):
            input_raw = cv2.imread(item, cv2.IMREAD_UNCHANGED) # [H, W, 3]
            input_raw = cv2.cvtColor(input_raw, cv2.COLOR_BGR2RGB) 
            roi = input_raw[y:y+f_len, x:x+f_len]
            rgbs.append(roi)
        count += 1
    
    rgbs = np.stack(rgbs, axis=0)
    imageio.mimwrite(os.path.join(path,name, f'{name}_25.mp4'), rgbs, fps=25, quality=8, macro_block_size=1)
    
    idx = 0
    count = 0
    depth_path = os.path.join(path,name,"depth_maps")
    make_clean_folder(depth_path)
    for item in tqdm.tqdm(depth_files):
        if(count%6 != 0 ):
            input_raw = cv2.imread(item, cv2.IMREAD_UNCHANGED) 
            roi = input_raw[y:y+f_len, x:x+f_len]
            cv2.imwrite(os.path.join(depth_path,f"{idx}.png"),roi)
            idx += 1
        count += 1
    


class Rgbd_recorder():
    def __init__(self,is_init = True, device=0) -> None:
        if is_init:
            self.config = o3d.io.AzureKinectSensorConfig() # 创建默认的配置对象
            self.sensor = o3d.io.AzureKinectSensor(self.config) # 创建传感器对象
            self.recorder = o3d.io.AzureKinectRecorder(self.config, device) # 创建记录器对象
            self.recorder.init_sensor() # 初始化传感器
        self.path = None
        self.name = None

    def extract_rgbd(self,path = None,name = None):
        if path != None and name != None:
            self.path = path
            self.name = name
        self.mkv_path = os.path.join(self.path,self.name,self.name + ".mkv")

        reader = o3d.io.AzureKinectMKVReader() # 创建阅读器
        reader.open(self.mkv_path) # 打开视频文件
        raw_data_rgb_path = os.path.join(self.path,self.name,"raw","color")
        raw_data_depth_path = os.path.join(self.path,self.name,"raw","depth")
        make_clean_folder(raw_data_rgb_path)
        make_clean_folder(raw_data_depth_path)

        idx = 0
        while not reader.is_eof(): # 判断视频是否全部读完
            rgbd = reader.next_frame() # 获取下一帧
            if(rgbd == None):
                continue
            color_filename = os.path.join(raw_data_rgb_path , '{0:05d}.jpg'.format(idx))
            # print('Writing to {}'.format(color_filename))
            o3d.io.write_image(color_filename, rgbd.color)
            depth_filename = os.path.join(raw_data_depth_path , '{0:05d}.png'.format(idx))
            # print('Writing to {}'.format(depth_filename))
            o3d.io.write_image(depth_filename, rgbd.depth)
            idx += 1



    # time:录制时长，单位s
    def run(self,path,name,times,pre_view = False):
        self.path = path
        self.name = name
        make_clean_folder(os.path.join(self.path,self.name))

        self.mkv_path = os.path.join(self.path,self.name,self.name + ".mkv")
        self.recorder.open_record(self.mkv_path) # 开启记录器
        num_frams = 30 * times
        if pre_view:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window('recorder', 1920, 540)
            vis_geometry_added = False
            for i in range(num_frams):
                rgbd = self.recorder.record_frame(enable_record = True, 
                                                    enable_align_depth_to_color = True)
                if rgbd is None or i%10 !=0:
                    continue
                if not vis_geometry_added:
                    vis.add_geometry(rgbd)
                    vis_geometry_added = True
                vis.update_geometry(rgbd)
                vis.poll_events()
                vis.update_renderer()
            self.recorder.close_record() 
        else:
            start = time.time()
            for i in range(num_frams): 
                rgbd = self.recorder.record_frame(enable_record = True, 
                                                    enable_align_depth_to_color = True)
            end = time.time()
            print(end-start)
            self.recorder.close_record() 
        


if __name__ == "__main__":
    r = Rgbd_recorder()
    r.run("data","wen",10)
    print("record finished")
    r.extract_rgbd("data","wen")
    crop_face("data","wen")