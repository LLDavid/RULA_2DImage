import numpy as np
import cv2
# from tensorflow.keras import layers
# import json
# from tensorflow.keras.models import model_from_json, load_model
import tf_pose
import pickle
import sys
sys.path.append("/home/lli40/PyCode/MyProject/RULA_2DImage/utils")
from utils import *

# # load reula estimator weights
# with open('model_architecture.json', 'r') as f:
#     model = model_from_json(f.read())
# model.load_weights('model_weights.h5')

def get_pose():
    # load video
    path_list = ["Sub04_03_30_FS_R11.avi", "Sub05_01_30_FK_R21.avi",
        "Sub05_02_30_KS_R11.avi", "Sub08_04_60_KS_R21.avi"]

    for i, img_path in enumerate(path_list):
        print(i)
        print(img_path)
        # dummy
        pose_all = []
        # video
        cap = cv2.VideoCapture(img_path)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # pose detection
                frame = frame[0:480, 50:480, :]
                coco_style = tf_pose.infer(frame)
                # check
                if len(coco_style)<1:
                    temp = np.ones((34,1))*-1.
                else:
                    temp = np.array(coco_style[0][0])
                    temp = np.expand_dims(np.delete(temp, range(2,51,3)), axis=1)

                temp=temp.tolist()
                pose_all.append(temp)
            else:
                break
            # print(i)
        np.save("./"+str(img_path[0:-4])+".npy", pose_all)
        # When everything done, release the capture
        cap.release()


def get_RULA():
    pose_list = ["Sub04_03_30_FS_R11.npy", "Sub05_01_30_FK_R21.npy",
        "Sub05_02_30_KS_R11.npy", "Sub08_04_60_KS_R21.npy"]
    for i, pose_path in enumerate(pose_list):
        pose = np.squeeze(np.load(pose_path))
        # load reula estimator weights
        with open('model_architecture.json', 'r') as f:
            model = model_from_json(f.read())
        model.load_weights('model_weights.h5')
        for ii in range(len(pose)):
            temp = np.expand_dims(pose[ii,:], axis=0)
            # temp = temp.tolist()
            # temp = pose[ii,:,:]
            # for t in range(34):
            #     if temp[0, t] == 0:
            #         temp[0, t] = -1
            outputs = model.predict(temp)
            predictions = np.argmax(outputs, axis=-1) + 1
            if predictions > 2:
                print(predictions)


def create_video():
    pose_list = ["./Sub04_01_30_FK_R11.npy", "./Sub05_01_30_FK_R21.npy",
                 "./Sub08_04_60_KS_R21.npy"]
    video_list = ["./Sub04_01_30_FK_R11.avi", "./Sub05_01_30_FK_R21.avi",
                 "./Sub08_04_60_KS_R21.avi"]
    for i, pose_path in enumerate(pose_list):
        pose = np.load(pose_path)
        cap = cv2.VideoCapture(video_list[i])

def get_video():
    # load video
    path_list = ["Sub04_01_30_FK_R11.avi", "Sub04_03_30_FS_R11.avi", "Sub04_02_30_KS_R21.avi",
                 "Sub05_01_30_FK_R21.avi", "Sub05_09_30_FS_R11.avi", "Sub05_02_30_KS_R11.avi",
                 "Sub08_01_00_FK_R11.avi,", "Sub08_11_30_FS_R11.avi", "Sub08_04_60_KS_R21.avi"]
    path_list=[ "Sub08_01_00_FS_R11.avi"]
    # load gscore
    gscore_name = np.loadtxt("/home/lli40/PyCode/MyProject/RULA_2DImage/data/gscore-name-lift.txt", dtype=str)
    gscore_name = np.delete(gscore_name, 0)
    for i, video_path in enumerate(path_list):
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        fid = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path[0:-4]+"_p.avi", fourcc, 20.0, (1440, 480))
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        ret = True
        while ret:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                print(str(fid)+" th frame")
                fid+=1
                # pose detection
                # frame = frame[0:480, 50:480, :]
                coco_style = tf_pose.infer(frame)
                # check
                if len(coco_style)<1:
                    temp = np.ones((34,1))*-1.
                else:
                    temp = np.array(coco_style[0][0])
                    print(temp.shape)
                visualizer = vis()
                new_frame = visualizer.vis_pose(image=frame, joints=temp)
                print(new_frame.shape)
                # cv2.imshow("a",new_frame)
                # cv2.waitKey(0)
                # exit()
                out.write(new_frame)
                # cv2.imshow("output", new_frame)
                # cv2.waitKey(0)
        cap.release()
        out.release()
        # cv2.distroyAllWindows()
def h36_video():
    img = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/pose100.npy", allow_pickle=True, encoding="latin1")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("H36_p.avi", fourcc, 30.0, (200, 150))
    for i in range(len(img)):
        frame = img[i] * 255.
        frame = frame.astype(np.uint8)
        coco_style = tf_pose.infer(frame)
        if len(coco_style) < 1:
            temp = np.ones((34, 1)) * -1.
        else:
            temp = np.array(coco_style[0][0])
            print(temp.shape)
            visualizer = vis()
            new_frame = visualizer.vis_pose(image=frame, joints=temp)
            print(new_frame.shape)
            for j in range(100):
                out.write(new_frame)
    out.release()
if __name__ == "__main__":
    h36_video()