from Thesis_Lib import *

# Driver code
# Object instantiation
obj_prepDataset = prepareDataset()

# Accessing class attributes
# and method through objects
print(obj_prepDataset.attr1)
# obj_prepDataset.forModelNet10_deleteBinvoxFiles()
obj_prepDataset.forModelNet10(shapeNumber= 10)


#
# #!/usr/bin/env python
# #-*-coding:utf-8-*-
#
# import os
# import numpy as np
# import binvox_rw
# import time
# from scipy.ndimage.interpolation import rotate
# from keras.utils.np_utils import to_categorical
#
#
# def findnamelistofclasses(dir):
#     os.chdir(dir + exdir)
#     currentdir = os.getcwd()
#     print "\nHangi directory'deyim"
#     print currentdir
#
#     # klasör isimlerini bul. Onlar ModelNet sınıf isimleridir.
#     listOfClasses = [d for d in os.listdir(currentdir) if os.path.isdir(os.path.join(currentdir, d))]
#     print "\nModelNet10 içindeki sınıf isimlerinin listesi"
#     print listOfClasses
#
#
#     #her sınıfa ait örnek sayılarını listele. minimum örnek sayısı olan sınıfı bul.
#     NumberOfMembers = list()
#     for classname in listOfClasses:
#         temp = directory + exdir + classname + "/train"
#         numberOfFile = len( os.listdir(temp) )
#         NumberOfMembers.append(numberOfFile)
#         print classname+"\t\tait örnek sayısı:\t", numberOfFile
#     minNumberOfMembers = min(NumberOfMembers)
#     print "\nMinNumberOfMembers:", minNumberOfMembers
#
#     return listOfClasses,NumberOfMembers,minNumberOfMembers
#
#
# def createarrayofsamples(directory,listOfClasses,NumberOfMembers,minNumberOfMembers,shapenumber):
#     totalsamplenumber = 0
#     i = 0
#     label = 0
#     datanumber = min(NumberOfMembers) * len(listOfClasses)
#     x_train = np.zeros(shape=(datanumber, shapenumber, shapenumber, shapenumber), dtype=float)
#     y_train = np.zeros((datanumber,), dtype="int")
#
#     os.chdir(directory)
#     currentdirectory = os.getcwd()
#     print "\nHangi directory'deyim"
#     print currentdirectory
#
#     for classname in listOfClasses:
#         path = directory + exdir + classname + "/train/"
#
#         k = 1
#         #while k <= 2:
#         while k <= minNumberOfMembers:
#             number = '{0:04}'.format(k)
#             print "**********************************************************", classname, "--", number  # , "         type(number):", type(number)
#             print "**********************************************************", "label: ", label
#             os.system(
#                 "./binvox -c -d " + str(shapenumber) + " -e -t binvox -cb " + path + classname + "_" + number + ".off")
#             binvoxfilepath = path + classname + "_" + number + ".binvox"
#
#             with open(binvoxfilepath, 'rb') as f:
#                 model = binvox_rw.read_as_3d_array(f)
#                 narray = np.ndarray(shape=(1,shapenumber, shapenumber, shapenumber), dtype=float)
#                 narray = model.data
#                 x_train[i, :] = narray.astype(float)
#                 y_train[i] = label
#                 totalsamplenumber = totalsamplenumber + 1
#                 i = i + 1
#             #time.sleep(2)
#             os.remove(binvoxfilepath)
#
#             k = k + 1
#         label = label + 1
#
#     np.save(directory + exdir + filename_x, x_train)
#     Y = to_categorical(y_train, len(listOfClasses))
#     print "Y.shape", Y.shape
#     np.save(directory + exdir + filename_y, Y)
#
#     file = open(directory + exdir + labelnames, "w")
#     for n in range(len(listOfClasses)):
#         file.write("%s " % (listOfClasses[n]))
#
#     # file.write(history.history['accuracy'])
#     # file.write("\nValidation Accuracy: ")
#     # file.write(history.history['val_accuracy'])
#     file.close()
#
#     print "totalsamplenumber: ", totalsamplenumber
#     print "\n***************** off to binvox --- binvox to numpy --- deleting binvox files --- save numpy files"
#     return x_train, Y
#
#
#
# def visualize_array():
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     import numpy as np
#
#     def press(event):
#         global x_train,count
#
#
#         if event.key == 'w':
#             count = count +1
#         if event.key == 'e':
#             count = count -1
#         print "User press: ", event.key, "numpy sample: ", count
#
#         plt.close()
#         one_of_x_train = x_train[count, :, :, :]
#         fig = plt.figure()
#         fig.canvas.mpl_connect('key_press_event', press)
#         ax = fig.gca(projection='3d')
#         ax.set_aspect('equal')
#         ax.voxels(one_of_x_train, edgecolor="k")
#         plt.show()
#
#         if event.key == "escape":
#             plt.close()
#
#
#     #y = np.load("etiket_matrisim.npy")
#     x_train = np.load(directory + exdir + filename_x)
#     print "x_train.shape: ", x_train.shape
#
#     print "Press 'W' for increment", "\nPress 'E' for decrement"
#
#     #count=0
#     print "User press: ", " ", "numpy sample: ", count
#     fig = plt.figure()
#     fig.canvas.mpl_connect('key_press_event', press)
#     ax = fig.gca(projection='3d')
#     ax.set_aspect('equal')
#     ax.voxels(x_train[count, :, :, :], edgecolor="k")
#     plt.show()
#
#
#
# ################################################# Main ###############
#
# #################################################################
# #bu kodun ve ModelNet klasörünün bulunduğu yer
# directory = "/home/enes/PycharmProjects/3DCNN_orginalfile/"
# exdir = "ModelNet10/ModelNet10/"
#
# filename_x = "x_train_10class_NoDataAugmentation_shape30_19August.npy"
# filename_y = "y_train_10class_NoDataAugmentation_shape30_19August.npy"
# labelnames = "list_of_labels__NoDataAugmentation_shape30_19August.txt"
#
# shapenumber = 30
# #################################################################
#
# start = time.time()
#
# listOfClasses,NumberOfMembers,minNumberOfMembers =     findnamelistofclasses(directory)
# x_train, y_train =    createarrayofsamples(directory,listOfClasses,NumberOfMembers,minNumberOfMembers,shapenumber)
#
# done = time.time()
# elapsed = done - start
#
# print "Elapsed time(seconds): ", elapsed
# print "\nModelNet10 içindeki sınıf isimlerinin listesi"
# print listOfClasses
# print "\nMinNumberOfMembers:", minNumberOfMembers
# print "\nShape of x_train:", x_train.shape
# print "\nShape of y_train:", y_train.shape
#
# count = 0
# visualize_array()