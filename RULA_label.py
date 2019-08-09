import numpy as np
import cv2

def Dis_Image(start_id=0, activity_id=0,file_dir="./",win_xy=(50,50)):
    """
    :param start_id: image to start with. e.g. start_id=i represents the i th image in that activity
    :param activity_id: activity to start with
    :param file_dir: directory for the image data
    :param win_xy: location for window display
    :return:
    """

    ## start from selected activity
    for i in range(activity_id, 15):
        if i <9: # be careful about the indexing !!!
            file_name = "ActivitySpecific_0"+str(i+1)+".npy"
            img_path = file_dir+file_name
            # annot_path = annot_dir + file_name
        else:
            file_name = "ActivitySpecific_" + str(i+1) + ".npy"
            img_path = file_dir + file_name
            # annot_path = annot_dir +file_name

        ## load pose
        pose_all=np.load(img_path)

        ## start from selected img
        for ii in range(start_id, len(pose_all)):
            img=pose_all[ii]
            win_name = "Activity ID: " + str(i+1) + " Image ID: " + str(ii)
            cv2.imshow(win_name, img)
            cv2.moveWindow(win_name, win_xy[0], win_xy[1])

            ## press q to quit; press any other key to next image
            if cv2.waitKey(0)==ord("q"):
                exit(0)
            else:
                cv2.destroyWindow(win_name)
                continue


if __name__ == "__main__":
    Dis_Image(start_id=2, activity_id=14,
              file_dir="/home/lli40/PyCode/MyProject/2DPose_detect/Data/Human80K/GTD2P/img_npy/", win_xy=(50, 50))