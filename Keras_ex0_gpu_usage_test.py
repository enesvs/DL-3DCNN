# buradaki çalışıyor
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# buradaki çalışıyor mu emin değilim
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()



# DEPTH_CAMERA = True
# import cv2
# if DEPTH_CAMERA:
#
#     all_camera_idx_available = []
#
#     for camera_idx in range(100):
#         cap = cv2.VideoCapture(camera_idx)
#         if cap.isOpened():
#             print(f'Camera index available: {camera_idx}')
#             all_camera_idx_available.append(camera_idx)
#             cap.release()
#
#     print("all_camera_idx_available: ", all_camera_idx_available)


    # # define a video capture object
    # vid = cv2.VideoCapture(0)
    #
    # while (True):
    #
    #     # Capture the video frame
    #     # by frame
    #     ret, frame = vid.read()
    #
    #     # Display the resulting frame
    #     cv2.imshow('frame', frame)
    #
    #     # the 'q' button is set as the
    #     # quitting button you may use any
    #     # desired button of your choice
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # After the loop release the cap object
    # vid.release()
    # # Destroy all the windows
    # cv2.destroyAllWindows()

