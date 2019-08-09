import numpy as np
import cv2
import sys
sys.path.append("/home/lli40/PyCode/Downloaded/tf-pose-estimation")
import tf_pose

class vis():

    def __init__(self):
        ## line pair
        ## index: nose-01, right eye-02, left eye-03, right ear-04, left ear-05,
        ##        right shoulder-06,  left shoulder-07, right elbow-08, left elbow-09,
        ##        right wrist -10, left wrist-11, right hip-12, left hip-13, right knee-14,
        ##        left knee-15, right ankle-16, left ankle-17
        self.lpair = np.array([[3, 5], [1 ,3], [1, 2], [2, 4], [6, 7], [7, 9], [9, 11], [6, 8],
               [8, 10], [7, 13], [6, 12], [12, 13], [13, 15], [15, 17], [12, 14], [14, 16]])-1

    def vis_pose(self, is_display = False, image=[], joints=[], is_ratio=False):
        assert len(joints) is not 0 and len(image) is not 0

        if is_ratio:
            w, h = img.shape
            joints[[jj for jj in range(0, 51, 3)]] *= w
            joints[[jj for jj in range(1, 51, 3)]] *= h

        joints = np.round(joints).astype(int)

        image = image.copy()

        img_shape = image.shape
        wimage = np.uint8(np.ones(img_shape)*255)

        ## draw lines
        for k in range(len(self.lpair)):
            color_no = np.uint8(np.random.randint(256, size=3))
            if joints[self.lpair[k, 0]*3] != 0 and joints[self.lpair[k, 1]*3] != 0:
                cv2.line(image, (joints[self.lpair[k, 0]*3], joints[self.lpair[k, 0]*3+1]),
                            (joints[self.lpair[k, 1]*3], joints[self.lpair[k, 1]*3+1]), color_no.tolist(), 2)
                cv2.line(wimage, (joints[self.lpair[k, 0] * 3], joints[self.lpair[k, 0] * 3 + 1]),
                         (joints[self.lpair[k, 1] * 3], joints[self.lpair[k, 1] * 3 + 1]), color_no.tolist(), 2)

        # draw key points
        for j in range(0, 51, 3):
            if joints[j] != 0 and joints[j + 1] != 0:
                cv2.circle(image, (joints[j], joints[j + 1]), 3, (0, 0, 255), -1)
                cv2.circle(wimage, (joints[j], joints[j + 1]), 3, (0, 0, 255), -1)

        # # draw rectangle on face
        # if joints[0] != 0 or joints[0] != -1:
        #     cv2.rectangle(image, (joints[0]-30, joints[1]-30), (joints[0]+5, joints[1]+30), (255, 255, 255), -1)
        ## display image
        if is_display:
            cv2.imshow("sample", np.hstack((image, wimage)))
            if cv2.waitKey(0) == ord('q'):
                 exit(0)
            else:
                print("Next")
        return np.hstack((image, wimage))


class io():

    def __init__(self):
        print(" ")

    def img2video(self, img_dir, video_dir):
        print(" ")

    def video2video(self, source_path, target_path):

        cap = cv2.VideoCapture(source_path)

        cnt= 0

        while (cap.isOpened()):

            cnt += 1

            ret, frame = cap.read()

            frame = frame[0:480, 50:480, :]

            coco_style = tf_pose.infer(frame)

            print(len(coco_style))

            joints = np.array(coco_style[0][0])

            visualizer = vis()

            new_frame = visualizer.vis_pose(image=frame, joints=joints)

            # new_frame = frame

            new_frame = new_frame[:, 200:-1, :]
            frame = frame [:, 200:-1, :]

            cv2.imwrite("./lift/" + str(cnt) + ".png", frame)
            cv2.imwrite("./lift/"+str(cnt)+"_n.png", new_frame)

            print(cnt)

            # cv2.imshow('frame', new_frame)
            #
            # if cv2.waitKey(0) == ord("q"):
            #     continue
            # else:
            #     exit()

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def img2img(self):
        img_path="/home/lli40/PyCode/MyProject/2DPose_detect/Data/Human80K/GTD2P/x_test.npy"
        img = np.load(img_path,  allow_pickle=True, encoding="latin1")
        cnt=0
        for i in range(len(img)):
            cnt+=1
            frame=img[i]*255.
            frame=frame.astype(np.uint8)
            coco_style = tf_pose.infer(frame)
            if len(coco_style)<1:
                continue
            joints = np.array(coco_style[0][0])
            cv2.imwrite("./human80/" + str(cnt) + ".png", frame)
            visualizer = vis()
            new_frame = visualizer.vis_pose(image=frame, joints=joints, is_display = False)
            cv2.imwrite("./human80/" + str(cnt) + "_n.png", new_frame)
            print(cnt)

class predict():
    def __init__(self):
        print("Initiated")

    def all_image(self, img_dir="/home/externalDisk/RULA_image/images_lift/side/"):

        from PIL import Image
        import glob
        image_list = []
        fname_list = []
        pose_list = []
        for i, file_path in enumerate(glob.glob(img_dir+"*.png")):
            print(i)
            ## append file name
            fname=file_path[47:]
            fname_list.append(fname)
            print(fname)
            ## append image
            im = np.array(Image.open(file_path))
            image_list.append(im)
            ## append predicted pose
            pose = tf_pose.infer(im)
            try:
                pose = np.array(pose[0][0])
            except:
                pose = np.zeros((51,))
            pose_list.append(pose)
        all3=[]
        all3.append(fname_list)
        all3.append(image_list)
        all3.append(pose_list)
        print(len(all3))

        import pickle

        with open("img_pose", "wb") as f:
            pickle.dump(all3, f)

    def test(self):

        import pickle
        visualizer = vis()

        with open('./img_pose', 'rb') as f:
            ...
            all3 = pickle.load(f)

            img_all = np.array(all3[1])
            pose_all = np.array(all3[2])

        for i in range(423):
            visualizer.vis_pose(is_display = True, image = img_all[i], joints = pose_all[i], is_ratio = False)




if __name__ == "__main__":

    aa = io()
    aa.img2img()
    # aa.video2video(source_path="/home/lli40/PyCode/MyProject/RULA_2DImage/utils/Sub01_15_30_FK_R21.avi", target_path="")