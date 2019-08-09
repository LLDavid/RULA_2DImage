import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def load_save_lift():
    # load pose
    with open("img_pose", 'rb') as f:
        all3 = pickle.load(f)
    print(all3[0][0])
    # load gscore
    gscore_name = np.loadtxt("/home/lli40/PyCode/MyProject/RULA_2DImage/data/gscore-name-lift.txt", dtype = str)
    gscore_name = np.delete(gscore_name, 0)
    # dummy
    pose_all=[]
    gscore_all=[]
    for i in range(len(gscore_name)):
        temp = gscore_name[i][0:-8]+".png"
        idx = all3[0].index(temp)
        temp1 = np.delete(all3[2][idx], range(2,51,3))
        pose_all.append(temp1)
        gscore_all.append(float(gscore_name[i][-1]))
    # to np array
    pose_all=np.array(pose_all)
    gscore_all=np.array(gscore_all)
    pose_all[pose_all==0]=-1
    print(pose_all.shape)
    print(pose_all[0])
    print(gscore_all.shape)


    # split
    x_train, x_test, y_train, y_test = train_test_split(pose_all, gscore_all, test_size=0.2, random_state=1)

    # # save
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_test_lift_raw.npy", x_test)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_train_lift_raw.npy", x_train)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_test_lift_raw.npy", y_test)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_train_lift_raw.npy", y_train)

    # # save all
    # np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_all.npy", pose_all)
    # np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_all.npy", gscore_all)

def load_save_h36():
    # load pose
    import sys
    sys.path.append("/home/lli40/PyCode/Downloaded/tf-pose-estimation")
    import tf_pose
    img_path = "/home/lli40/PyCode/MyProject/RULA_2DImage/data/pose100.npy"
    img = np.load(img_path, allow_pickle=True, encoding="latin1")
    cnt = 0
    pose_all=[]
    # import time
    for i in range(len(img)):
        # start_time = time.time()
        cnt += 1
        frame = img[i] * 255.
        frame = frame.astype(np.uint8)
        coco_style = tf_pose.infer(frame)
        # elapsed_time = time.time() - start_time
        # print("FPS: " + str(1. / elapsed_time))
        if len(coco_style)<1:
            temp = np.ones((34,1))*-1.
        else:
            temp = np.array(coco_style[0][0])
            temp = np.expand_dims(np.delete(temp, range(2,51,3)), axis=1)
        print(temp.shape)
        temp=temp.tolist()
        pose_all.append(temp)

    exit()
    pose_all=np.array(pose_all)
    print(pose_all.shape)

    # load gscore
    gscore_name = np.loadtxt("/home/lli40/PyCode/MyProject/RULA_2DImage/data/gscore-name-human36.txt", dtype = str)
    gscore_name = np.delete(gscore_name, 0)
    # dummy
    gscore_all=[]
    for i in range(len(gscore_name)):
        idx=gscore_name[i][0:-3]
        temp=gscore_name[i][-1]
        gscore_all.append(int(temp))
    # to np array
    gscore_all=np.array(gscore_all)

    # save
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_all_h36_raw.npy", pose_all)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_all_h36_raw.npy", gscore_all)

def data_aug_lift():
    ## init
    pose_ori = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_all.npy")
    gscore_ori = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_all.npy")
    cnt = 194./np.array([12, 194, 56, 80, 38, 27, 16])+0.99

    ## new
    pose_aug=[]
    gscore_aug=[]

    import random
    for i in range(len(pose_ori)):
        pose_aug.append(pose_ori[i,:])
        gscore_aug.append(gscore_ori[i])
        for j in range(int((cnt[int(gscore_ori[i])-1]))):
            temp0 = random.choice(range(16))
            temp1 = pose_ori[i,:]
            temp1[temp0*2]=-1
            temp1[temp0*2+1]=-1
            for ii in range(len(temp1)):
                if temp1[ii] != -1:
                    temp1[ii] = temp1[ii]+int(np.random.normal(0, 2, 1))
            pose_aug.append(temp1)
            gscore_aug.append(gscore_ori[i])


    for i in range(7):
        print(i+1)
        print(np.sum(np.array(gscore_aug)==(i+1)))
    # split
    x_train, x_test, y_train, y_test = train_test_split(pose_aug, gscore_aug, test_size=0.2, random_state=1)

    # save
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_test_lift_aug.npy", x_test)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_train_lift_aug.npy", x_train)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_test_lift_aug.npy", y_test)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_train_lift_aug.npy", y_train)

    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_all_lift_aug.npy", pose_aug)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_all_lift_aug.npy", gscore_aug)

def data_aug_h36():
    ## init
    pose_ori = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_all_h36_raw.npy")
    gscore_ori = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_all_h36_raw.npy")
    cnt = np.array([0, 100/10, 100/48, 100/16, 100/14, 100/10, 100/2])
    ## new
    pose_aug=[]
    gscore_aug=[]

    import random
    for i in range(len(pose_ori)):
        pose_aug.append(pose_ori[i])
        gscore_aug.append(gscore_ori[i])
        for j in range(int((cnt[int(gscore_ori[i])-1]))):
            temp0 = random.choice(range(16))
            temp1 = pose_ori[i]
            temp1[temp0*2]=-1
            temp1[temp0*2+1]=-1
            for ii in range(len(temp1)):
                if temp1[ii] != -1:
                    temp1[ii] = temp1[ii]+int(np.random.normal(0,2,1))
                    print("ran:"+str(int(np.random.normal(0,2,1))))
            pose_aug.append(temp1)
            gscore_aug.append(gscore_ori[i])


    for i in range(7):
        print(i+1)
        print(np.sum(np.array(gscore_aug)==(i+1)))
    # split
    x_train, x_test, y_train, y_test = train_test_split(pose_aug, gscore_aug, test_size=0.2, random_state=1)

    # save
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_test_h36_aug.npy", x_test)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_train_h36_aug.npy", x_train)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_test_h36_aug.npy", y_test)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_train_h36_aug.npy", y_train)

    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_all_h36_aug.npy", pose_aug)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_all_h36_aug.npy", gscore_aug)

def merge2():
    x_train1 = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_train_lift_aug.npy")
    x_test1 = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_test_lift_aug.npy")
    y_train1 = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_train_lift_aug.npy")
    y_test1 = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_test_lift_aug.npy")
    x_train2 = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_train_h36_aug.npy")
    x_test2 = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_test_h36_aug.npy")
    y_train2 = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_train_h36_aug.npy")
    y_test2 = np.load("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_test_h36_aug.npy")


    x_train = np.vstack((x_train1, np.squeeze(x_train2, axis=2)))
    x_test = np.vstack((x_test1, np.squeeze(x_test2, axis=2)))
    y_train = np.vstack((np.expand_dims(y_train1, axis=1),np.expand_dims(y_train2,axis=1)))
    y_test = np.vstack((np.expand_dims(y_test1, axis=1), np.expand_dims(y_test2, axis=1)))

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_test.npy", x_test)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/x_train.npy", x_train)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_test.npy", y_test)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data/y_train.npy", y_train)


def Indices2OneHot(class_indices):
    class_indices = class_indices.astype(int)
    max_i=np.max(class_indices)+1
    class_labels=np.zeros([np.size(class_indices,0),max_i])
    for i in range(np.size(class_indices,0)):
        class_labels[i][class_indices[i]]=1
    #class_indices = class_indices.astype(int)
    return class_labels

def data_upload_lift():
    ## for lift
    lift_img_dir="/home/externalDisk/RULA_image/images_lift/side/"
    from PIL import Image
    import glob
    image_list = []
    fname_list = []

    import cv2
    import sys
    sys.path.append("/home/lli40/PyCode/Downloaded/tf-pose-estimation")
    import tf_pose

    for i, file_path in enumerate(glob.glob(lift_img_dir + "*.png")):
        print(i)
        ## append file name
        fname = file_path[47:]
        fname_list.append(fname)
        print(fname)
        ## append image
        im = np.array(Image.open(file_path))
        image = im[0:480, 50:480, :]

        coco_style = tf_pose.infer(image)

        if len(coco_style)>0:

            joints = np.array(coco_style[0][0])

            joints = np.round(joints).astype(int)

            image = image.copy()

            # draw rectangle on face
            if joints[0] != 0 or joints[0] != -1:
                cv2.rectangle(image, (joints[0]-30, joints[1]-30), (joints[0]+5, joints[1]+30), (255, 255, 255), -1)

        image_list.append(image)

        # cv2.imshow("sample", image)
        # cv2.waitKey(0)

    print(np.array(image_list).shape)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data_upload/lift_image.npy", image_list)

def data_upload_h36():
    img_path = "/home/lli40/PyCode/MyProject/RULA_2DImage/data/pose100.npy"
    a = np.load(img_path)
    np.save("/home/lli40/PyCode/MyProject/RULA_2DImage/data_upload/h36_image.npy", a)

if __name__ == "__main__":
    # data_aug()
    # load_save_h36()
    # load_save()
    # load_save_lift()
    # data_aug_lift()
    # data_aug_h36()
    # merge2()
    data_upload_h36()