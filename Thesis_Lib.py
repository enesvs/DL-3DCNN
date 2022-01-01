import os
import numpy as np
import binvox_rw
import datetime
import time

from sklearn.model_selection import train_test_split

from tensorflow.keras import Model, initializers, activations
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten , GaussianNoise
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.nn import relu, softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model, to_categorical


class Dataset:

    def __init__(self, prefixDir):
        print("\n----- Dataset Initialization  started!! -----")
        time.sleep(2)

        self.prefixDir = prefixDir

        self.classesInfo = {"listOfClasses": list(),
                            "numberOfSample_in_trainFolder": dict(),
                            "minNumberOfSample_in_trainFolder": 0,
                            "nameListOfSamples_in_trainFolder": dict(),
                            "sampleNameList_in_trainFolder" : dict(),

                            "numberOfSample_in_testFolder": dict(),
                            "minNumberOfSample_in_testFolder": 0,
                            "nameListOfSamples_in_testFolder": dict(),
                            "sampleNameList_in_testFolder": dict(),
                            }

        for file in os.listdir(self.prefixDir):
            if os.path.isdir(os.path.join(self.prefixDir, file)):
                self.classesInfo["listOfClasses"].append(file)

        self.classesInfo["listOfClasses"].sort()



        for classname in self.classesInfo["listOfClasses"]:
            temp = list()
            for fileName in os.listdir(self.prefixDir + classname + "/train"):
                if ".off" in fileName:
                    temp.append(fileName)

            temp.sort()
            self.classesInfo["sampleNameList_in_trainFolder"][classname] = temp
            self.classesInfo["numberOfSample_in_trainFolder"][classname] = len(temp)
            # print("asdasd ", self.classesInfo["sampleNameList_in_trainFolder"][classname])



        for classname in self.classesInfo["listOfClasses"]:
            temp = list()
            for fileName in os.listdir(self.prefixDir + classname + "/test"):
                if ".off" in fileName:
                    temp.append(fileName)

            temp.sort()
            self.classesInfo["sampleNameList_in_testFolder"][classname] = temp
            self.classesInfo["numberOfSample_in_testFolder"][classname] = len(temp)

        self.classesInfo["minNumberOfSample_in_trainFolder"] = min(self.classesInfo["numberOfSample_in_trainFolder"].values())
        self.classesInfo["minNumberOfSample_in_testFolder"] = min(self.classesInfo["numberOfSample_in_testFolder"].values())

        firstClassName_in_trainFolder = self.classesInfo["listOfClasses"][0]

        print("listOfClasses:", self.classesInfo["listOfClasses"],
              "\nminNumberOfSample_in_trainFolder", self.classesInfo["minNumberOfSample_in_trainFolder"],
              "\nminNumberOfSample_in_testFolder", self.classesInfo["minNumberOfSample_in_testFolder"],
              "\nsampleNameList_in_trainFolder", self.classesInfo["sampleNameList_in_trainFolder"][ classname ][0:10],
              "\nsampleNameList_in_testFolder", self.classesInfo["sampleNameList_in_testFolder"][ classname ][0:10])

        print("----- Dataset Initialization  finished!! -----\n")

    def deleteBinvoxFiles(self):
        print("----- deleteBinvoxFiles  started!!")
        time.sleep(2)

        totalCounter = 0

        for classname in self.classesInfo["listOfClasses"]:
            fileNames_in_trainFolder = os.listdir(self.prefixDir + classname + "/train")
            # print("TrainFolder classname:", classname, "list: ", sampleNameList_inTrainFolder)
            for fileName in fileNames_in_trainFolder:
                if ".binvox" in fileName:
                    os.remove(self.prefixDir + classname + "/train" + "/" + fileName)
                    print("'{}' is removing..".format(fileName))
                    totalCounter = totalCounter + 1

            fileNames_in_testFolder = os.listdir(self.prefixDir + classname + "/test")
            # print("TrainFolder classname:", classname, "list: ", sampleNameList_inTrainFolder)
            for fileName in fileNames_in_testFolder:
                if ".binvox" in fileName:
                    os.remove(self.prefixDir + classname + "/test" + "/" + fileName)
                    print("'{}' is removing..".format(fileName))
                    totalCounter = totalCounter + 1

        print("{} files were deleted.".format(totalCounter))
        print("----- deleteBinvoxFiles  finished!!")

    def create_binvoxFiles_from_offFiles(self, shapeNumber):
        print("\n----- create_binvoxFiles_from_offFiles  started!!")
        time.sleep(2)

        totalCounter = 0

        minNumber_train = self.classesInfo["minNumberOfSample_in_trainFolder"]
        for classname in self.classesInfo["listOfClasses"]:
            cl_name = classname
            for sampleName in self.classesInfo["sampleNameList_in_trainFolder"][cl_name][0:minNumber_train]:
                os.system(
                    "./binvox -c -d " + str(
                        shapeNumber) + " -e -t binvox -cb " + self.prefixDir + classname + "/train" + "/" + sampleName)
                totalCounter = totalCounter + 1

        minNumber_test = self.classesInfo["minNumberOfSample_in_testFolder"]
        for classname in self.classesInfo["listOfClasses"]:
            cl_name = classname
            for sampleName in self.classesInfo["sampleNameList_in_testFolder"][cl_name][0:minNumber_test]:

                os.system(
                    "./binvox -c -d " + str(
                        shapeNumber) + " -e -t binvox -cb " + self.prefixDir + classname + "/test" + "/" + sampleName)
                totalCounter = totalCounter + 1

        print("{} files were created.".format(totalCounter))
        print("----- create_binvoxFiles_from_offFiles  finished!!")


    # def forModelNet10_deleteBinvoxFiles(self):
    #     print("----- prepareDataset --> forModelNet10_deleteBinvoxFiles  started!!")
    #
    #     currentdir = os.getcwd()
    #     ClassDir_in_ModelNet = currentdir + "/Datasets/" + "ModelNet10/ModelNet10/"
    #     print("ModelNet içindeki tüm class folder'lar burada olmalı : ", ClassDir_in_ModelNet)
    #
    #     # klasör isimlerini bul. Onlar ModelNet sınıf isimleridir.
    #     ListOfClasses = [d for d in os.listdir(ClassDir_in_ModelNet) if
    #                      os.path.isdir(os.path.join(ClassDir_in_ModelNet, d))]
    #     # print("\nModelNet10 içindeki sınıf isimlerinin listesi :", ListOfClasses)
    #
    #     removedFilesCounter = 0
    #
    #     # her sınıfa ait örnek sayılarını listele. minimum örnek sayısı olan sınıfı bul.
    #     NumberOfSamples_in_theClass = list()
    #     for classname in ListOfClasses:
    #         sampleNameList_inTrainFolder = os.listdir(ClassDir_in_ModelNet + classname + "/train")
    #         # print("classname:", classname, "list: ", sampleNameList)
    #
    #         for sampleName in sampleNameList_inTrainFolder:
    #             if ".binvox" in sampleName:
    #                 os.remove(ClassDir_in_ModelNet + classname + "/train" + "/" + sampleName)
    #                 # print("'{}' is removing..".format(sampleName))
    #                 removedFilesCounter = removedFilesCounter + 1
    #
    #         sampleNameList_inTestFolder = os.listdir(ClassDir_in_ModelNet + classname + "/test")
    #         for sampleName in sampleNameList_inTestFolder:
    #             if ".binvox" in sampleName:
    #                 os.remove(ClassDir_in_ModelNet + classname + "/test" + "/" + sampleName)
    #                 # print("'{}' is removing..".format(sampleName))
    #                 removedFilesCounter = removedFilesCounter +1
    #
    #
    #     print("{} binvox files are removed in the dataset including 10 class!!".format(removedFilesCounter))
    #
    #     print("----- prepareDataset --> forModelNet10_deleteBinvoxFiles  finished!!")

    def forModelNet10_create_binvoxFiles_from_offFiles(self, shapeNumber):
        print("----- prepareDataset --> forModelNet10_create_binvoxFiles_from_offFiles  started!!")

        currentdir = os.getcwd()
        ClassDir_in_ModelNet = currentdir + "/Datasets/" + "ModelNet10/ModelNet10/"
        print("ModelNet içindeki tüm class folder'lar burada olmalı : ", ClassDir_in_ModelNet)


        # klasör isimlerini bul. Onlar ModelNet sınıf isimleridir.
        ListOfClasses = [d for d in os.listdir(ClassDir_in_ModelNet) if os.path.isdir(os.path.join(ClassDir_in_ModelNet, d))]
        # print("\nModelNet10 içindeki sınıf isimlerinin listesi :", ListOfClasses)


        #### For Train Folder
        # her sınıfa ait örnek sayılarını listele. minimum örnek sayısı olan sınıfı bul.
        NumberOfSamples_in_theClass_trainFolder = list()
        for classname in ListOfClasses:
            numberOfSample_in_TrainFolder = len(os.listdir(ClassDir_in_ModelNet + classname + "/train"))
            NumberOfSamples_in_theClass_trainFolder.append(numberOfSample_in_TrainFolder)

        # print("NumberOfSamples_in_theClass:\t\t\t\t\t", NumberOfSamples_in_theClass)

        print("\nClassname -- NumberOfSamples")
        for count in range(len(ListOfClasses)):
            print("{}:\t\t{}".format(ListOfClasses[count], NumberOfSamples_in_theClass_trainFolder[count]))

        minNumberOfSamples_in_trainFolder = min(NumberOfSamples_in_theClass_trainFolder)
        print("\nminNumberOfSamples_in_trainFolder:", minNumberOfSamples_in_trainFolder)


        for nClass, classname in enumerate(ListOfClasses):
            print("************* {}   {}/{}".format(classname, nClass+1, len(ListOfClasses)) )
            for nSample in range(1,minNumberOfSamples_in_trainFolder+1):

                number = '{0:04}'.format(nSample)
                # print("*************", classname, "--", number,"         type(number):", type(number))
                # print("*************", "label: ", label)

                # print("converting off file to binvox file -- {}".format(
                #     ClassDir_in_ModelNet + classname + "/train" + "/" + classname + "_" + number + ".off"))
                # !!! write "chmod 755 binvox" in the terminal in the directory which binvox existed in.
                os.system(
                    "./binvox -c -d " + str(
                        shapeNumber) + " -e -t binvox -cb " + ClassDir_in_ModelNet + classname + "/train" + "/" + classname + "_" + number + ".off")

        # print("\n\n*************", classname, "  nClass", nClass)
        # print("*************", nSample)
        print("\nminNumberOfSamples_in_trainFolder:", minNumberOfSamples_in_trainFolder)

        print("----- prepareDataset --> forModelNet10_create_binvoxFiles_from_offFiles  finished!!")

        return minNumberOfSamples

    def forModelNet10_create_numpyFiles_from_binvoxFiles(self, shapeNumber, minNumberOfSamples):
        print("----- prepareDataset --> forModelNet10_create_numpyFiles_from_binvoxFiles  started!!")

        currentdir = os.getcwd()
        ClassDir_in_ModelNet = currentdir + "/Datasets/" + "ModelNet10/ModelNet10/"
        outputFiles_in_ModelNet10 = currentdir + "/Datasets/" + "ModelNet10_output/"
        print("ModelNet içindeki tüm class folder'lar burada olmalı : ", ClassDir_in_ModelNet)

        # klasör isimlerini bul. Onlar ModelNet sınıf isimleridir.
        ListOfClasses = [d for d in os.listdir(ClassDir_in_ModelNet) if
                         os.path.isdir(os.path.join(ClassDir_in_ModelNet, d))]
        # print("\nModelNet10 içindeki sınıf isimlerinin listesi :", ListOfClasses)

        # her sınıfa ait örnek sayılarını listele. minimum örnek sayısı olan sınıfı bul.
        NumberOfSamples_in_theClass = list()
        for classname in ListOfClasses:
            numberOfSample_in_TrainFolder = len(os.listdir(ClassDir_in_ModelNet + classname + "/train"))
            NumberOfSamples_in_theClass.append(numberOfSample_in_TrainFolder)

        # print("NumberOfSamples_in_theClass:\t\t\t\t\t", NumberOfSamples_in_theClass)

        # print("\nClassname -- NumberOfSamples")
        # for count in range(len(ListOfClasses)):
        #     print("{}:\t\t{}".format(ListOfClasses[count], NumberOfSamples_in_theClass[count]))

        # minNumberOfSamples = min(NumberOfSamples_in_theClass)
        print("\nMinNumberOfMembers:", minNumberOfSamples)




        totalsamplenumber = 0
        trainSamples_counter = 0
        label = 0
        dataNumber = minNumberOfSamples * len(ListOfClasses)
        x_train = np.zeros(shape=(dataNumber, shapeNumber, shapeNumber, shapeNumber), dtype=float)
        y_train = np.zeros((dataNumber,), dtype="int")




        for nClass, classname in enumerate(ListOfClasses):
            print("************* {}   {}/{}".format(classname, nClass+1, len(ListOfClasses)))
            for nSample in range(1,minNumberOfSamples+1):
                number = '{0:04}'.format(nSample)

            binvoxfilepath = ClassDir_in_ModelNet + classname + "/train" + "/" + classname + "_" + number + ".binvox"

            with open(binvoxfilepath, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)
                narray = np.ndarray(shape=(1, shapeNumber, shapeNumber, shapeNumber), dtype=float)
                narray = model.data
                x_train[trainSamples_counter, :] = narray.astype(float)
                y_train[trainSamples_counter] = label
                totalsamplenumber = totalsamplenumber + 1
                trainSamples_counter = trainSamples_counter + 1
            # time.sleep(2)
            # os.remove(binvoxfilepath)

            label = label + 1

        dt = datetime.datetime.today()

        print("x_train.shape", x_train.shape)
        np.save(outputFiles_in_ModelNet10 + "x_train_shapeNumber{}__{}_{}_{}__{}_{}_{}.npy".format(shapeNumber, dt.day, dt.month, dt.year, dt.hour, dt.minute, dt.second) , x_train)

        y_train_categorical = to_categorical(y_train, len(ListOfClasses))
        print("y_train_categorical.shape", y_train_categorical.shape)
        np.save(outputFiles_in_ModelNet10 + "y_train_shapeNumber{}_{}_{}_{}__{}_{}_{}.npy".format(shapeNumber, dt.day, dt.month, dt.year, dt.hour, dt.minute, dt.second), y_train_categorical)


        print("----- prepareDataset --> forModelNet10_create_numpyFiles_from_binvoxFiles  finished!!")



    def forModelNet40_deleteBinvoxFiles(self):
        print("----- prepareDataset --> forModelNet40_deleteBinvoxFiles  started!!")

        currentdir = os.getcwd()
        ClassDir_in_ModelNet = currentdir + "/Datasets/" + "ModelNet40/ModelNet40/"
        print("ModelNet içindeki tüm class folder'lar burada olmalı : ", ClassDir_in_ModelNet)

        # klasör isimlerini bul. Onlar ModelNet sınıf isimleridir.
        ListOfClasses = [d for d in os.listdir(ClassDir_in_ModelNet) if
                         os.path.isdir(os.path.join(ClassDir_in_ModelNet, d))]
        # print("\nModelNet10 içindeki sınıf isimlerinin listesi :", ListOfClasses)

        removedFilesCounter = 0

        # her sınıfa ait örnek sayılarını listele. minimum örnek sayısı olan sınıfı bul.
        NumberOfSamples_in_theClass = list()
        for classname in ListOfClasses:
            sampleNameList = os.listdir(ClassDir_in_ModelNet + classname + "/train")
            # print("classname:", classname, "list: ", sampleNameList)

            for sampleName in sampleNameList:
                if ".binvox" in sampleName:
                    os.remove(ClassDir_in_ModelNet + classname + "/train" + "/" + sampleName)
                    # print("'{}' is removing..".format(sampleName))
                    removedFilesCounter = removedFilesCounter +1


        print("{} binvox files are removed in the dataset including 40 class!!".format(removedFilesCounter))

        print("----- prepareDataset --> forModelNet40_deleteBinvoxFiles  finished!!")

    def forModelNet40_create_binvoxFiles_from_offFiles(self, shapeNumber):
        print("----- prepareDataset --> forModelNet40_create_binvoxFiles_from_offFiles  started!!")

        currentdir = os.getcwd()
        ClassDir_in_ModelNet = currentdir + "/Datasets/" + "ModelNet40/ModelNet40/"
        print("ModelNet içindeki tüm class folder'lar burada olmalı : ", ClassDir_in_ModelNet)


        # klasör isimlerini bul. Onlar ModelNet sınıf isimleridir.
        ListOfClasses = [d for d in os.listdir(ClassDir_in_ModelNet) if os.path.isdir(os.path.join(ClassDir_in_ModelNet, d))]
        # print("\nModelNet10 içindeki sınıf isimlerinin listesi :", ListOfClasses)

        # her sınıfa ait örnek sayılarını listele. minimum örnek sayısı olan sınıfı bul.
        NumberOfSamples_in_theClass = list()
        for classname in ListOfClasses:
            numberOfSample_in_TrainFolder = len(os.listdir(ClassDir_in_ModelNet + classname + "/train"))
            NumberOfSamples_in_theClass.append(numberOfSample_in_TrainFolder)

        # print("NumberOfSamples_in_theClass:\t\t\t\t\t", NumberOfSamples_in_theClass)

        print("\nClassname -- NumberOfSamples")
        for count in range(len(ListOfClasses)):
            print("{}:\t\t{}".format(ListOfClasses[count], NumberOfSamples_in_theClass[count]))

        minNumberOfSamples = min(NumberOfSamples_in_theClass)
        print("\nMinNumberOfMembers:", minNumberOfSamples)



        # totalsamplenumber = 0
        # trainSamples_counter = 0
        # label = 0
        # dataNumber = minNumberOfSamples * len(ListOfClasses)
        # x_train = np.zeros(shape=(dataNumber, shapeNumber, shapeNumber, shapeNumber), dtype=float)
        # y_train = np.zeros((dataNumber,), dtype="int")



        for nClass, classname in enumerate(ListOfClasses):
            print("************* {}   {}/{}".format(classname, nClass+1, len(ListOfClasses)) )
            for nSample in range(1,minNumberOfSamples+1):

                number = '{0:04}'.format(nSample)
                # print("*************", classname, "--", number,"         type(number):", type(number))
                # print("*************", "label: ", label)

                # print("converting off file to binvox file -- {}".format(
                #     ClassDir_in_ModelNet + classname + "/train" + "/" + classname + "_" + number + ".off"))
                # !!! write "chmod 755 binvox" in the terminal in the directory which binvox existed in.
                os.system(
                    "./binvox -c -d " + str(
                        shapeNumber) + " -e -t binvox -cb " + ClassDir_in_ModelNet + classname + "/train" + "/" + classname + "_" + number + ".off")
                # time.sleep(2)

        # print("\n\n*************", classname, "  nClass", nClass)
        # print("*************", nSample)
        print("\nMinNumberOfMembers:", minNumberOfSamples)

        print("----- prepareDataset --> forModelNet40_create_binvoxFiles_from_offFiles  finished!!")

        return minNumberOfSamples

    def forModelNet40_create_numpyFiles_from_binvoxFiles(self, shapeNumber, minNumberOfSamples):
        print("----- prepareDataset --> forModelNet40_create_numpyFiles_from_binvoxFiles  started!!")

        currentdir = os.getcwd()
        ClassDir_in_ModelNet = currentdir + "/Datasets/" + "ModelNet40/ModelNet40/"
        outputFiles_in_ModelNet40 = currentdir + "/Datasets/" + "ModelNet40_output/"
        print("ModelNet içindeki tüm class folder'lar burada olmalı : ", ClassDir_in_ModelNet)

        # klasör isimlerini bul. Onlar ModelNet sınıf isimleridir.
        ListOfClasses = [d for d in os.listdir(ClassDir_in_ModelNet) if
                         os.path.isdir(os.path.join(ClassDir_in_ModelNet, d))]
        print("\nModelNet10 içindeki sınıf isimlerinin listesi :", ListOfClasses)

        # her sınıfa ait örnek sayılarını listele. minimum örnek sayısı olan sınıfı bul.
        NumberOfSamples_in_theClass = list()
        for classname in ListOfClasses:
            numberOfSample_in_TrainFolder = len(os.listdir(ClassDir_in_ModelNet + classname + "/train"))
            NumberOfSamples_in_theClass.append(numberOfSample_in_TrainFolder)

        # print("NumberOfSamples_in_theClass:\t\t\t\t\t", NumberOfSamples_in_theClass)

        # print("\nClassname -- NumberOfSamples")
        # for count in range(len(ListOfClasses)):
        #     print("{}:\t\t{}".format(ListOfClasses[count], NumberOfSamples_in_theClass[count]))

        # minNumberOfSamples = min(NumberOfSamples_in_theClass)
        print("\nMinNumberOfMembers:", minNumberOfSamples)




        totalsamplenumber = 0
        trainSamples_counter = 0
        label = 0
        dataNumber = minNumberOfSamples * len(ListOfClasses)
        x_train = np.zeros(shape=(dataNumber, shapeNumber, shapeNumber, shapeNumber), dtype=float)
        y_train = np.zeros((dataNumber,), dtype="int")




        for nClass, classname in enumerate(ListOfClasses):
            print("************* {}   {}/{}".format(classname, nClass+1, len(ListOfClasses)))
            for nSample in range(1,minNumberOfSamples+1):
                number = '{0:04}'.format(nSample)

            binvoxfilepath = ClassDir_in_ModelNet + classname + "/train" + "/" + classname + "_" + number + ".binvox"

            with open(binvoxfilepath, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)
                narray = np.ndarray(shape=(1, shapeNumber, shapeNumber, shapeNumber), dtype=np.uint8)
                narray = model.data
                x_train[trainSamples_counter, :] = narray.astype(np.uint8)
                y_train[trainSamples_counter] = label
                totalsamplenumber = totalsamplenumber + 1
                trainSamples_counter = trainSamples_counter + 1
            # time.sleep(2)
            # os.remove(binvoxfilepath)

            label = label + 1

        dt = datetime.datetime.today()

        print("x_train.shape", x_train.shape)
        np.save(outputFiles_in_ModelNet40 + "x_train_shapeNumber{}_{}_{}_{}__{}_{}_{}.npy".format(shapeNumber, dt.day, dt.month, dt.year, dt.hour, dt.minute, dt.second) , x_train)

        y_train_categorical = to_categorical(y_train, len(ListOfClasses))
        print("y_train_categorical.shape", y_train_categorical.shape)
        np.save(outputFiles_in_ModelNet40 + "y_train_shapeNumber{}_{}_{}_{}__{}_{}_{}.npy".format(shapeNumber, dt.day, dt.month, dt.year, dt.hour, dt.minute, dt.second), y_train_categorical)


        print("----- prepareDataset --> forModelNet40_create_numpyFiles_from_binvoxFiles  finished!!")



class TrainModel:

    def __init__(self, prefixDir):
        self.prefixDir = prefixDir

        dt = datetime.datetime.today()
        self.prefixForOutputFile = "output_{}{}{}__{}_{}_{}__".format(dt.day, dt.month, dt.year,
                                                                 dt.hour, dt.minute, dt.second)

    def load_npyFile(self, filename__x_train, filename__y_train):
        print("----- TrainModel --> load_npyFile  started!!")

        y_total = np.load(filename__y_train)
        # y_total = y_total.astype(float)
        x_total = np.load(filename__x_train)
        x_total = x_total.astype(float)
        x_total = np.expand_dims(x_total, axis=4)
        print("x_total shape: ", x_total.shape, "\ty_total shape: ", y_total.shape)

        x_train, x_valid, y_train, y_valid = train_test_split(x_total, y_total, test_size=0.2, random_state=1)
        print("\nx_train shape: ", x_train.shape, "\ty_train shape: ", y_train.shape)
        print("x_valid shape: ", x_valid.shape, "\ty_valid shape: ", y_valid.shape)


        print("----- TrainModel --> load_npyFile  finished!!")

        return x_train, x_valid, y_train, y_valid

    def train_with_VoxNet(self, x_train, x_valid, y_train, y_valid):
        print("----- TrainModel --> train_with_VoxNet  started!!")

        #### Nested definiton START ####
        def my_initializer(shape, dtype=None):
            receptive_field_size = np.prod(shape[2:])
            c = shape[1]  # input channels
            nl = c * receptive_field_size
            std = np.sqrt(2.0 / (nl))
            return np.random.normal(0, std, size=shape).astype('float32')
        #### Nested definiton END   ####


        input_shape = x_train.shape[1:5]
        num_classes = y_train.shape[1]
        print("input shape: ", input_shape, "num classes: ", num_classes)

        inputs1 = Input(shape=input_shape)

        layer1 = Conv3D(8, 3, strides=(2, 2, 2), padding="valid",
                        activation='relu', kernel_initializer=my_initializer)(inputs1)
        layer1 = Dropout(0.2)(layer1)

        layer2 = Conv3D(8, 3, strides=(2, 2, 2), padding="valid",
                        activation='relu', kernel_initializer=my_initializer)(layer1)
        layer2 = MaxPooling3D(pool_size=(2, 2, 2), padding="valid")(layer2)
        layer2 = Dropout(0.3)(layer2)

        flat = Flatten()(layer2)

        dense1 = Dense(128, kernel_initializer=initializers.RandomNormal(stddev=0.01))(flat)
        dense1 = Dropout(0.4)(dense1)

        outputs = Dense(num_classes, kernel_initializer=initializers.RandomNormal(stddev=0.01))(dense1)
        outputs = Activation("softmax")(outputs)

        model = Model(inputs=inputs1, outputs=outputs)
        model.summary()

        filepath = self.prefixDir + self.prefixForOutputFile + "Architecture.png"
        ## ""sudo apt-get install graphviz""
        ## terminal'e bunu yazmalısın. virEnv bağımsız yüklemelisin.
        plot_model(model, show_shapes=True, to_file=filepath)

        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid),
                            epochs=10, batch_size=16, verbose=2)

        model_filepath = self.prefixDir + self.prefixForOutputFile + "Model.h5"
        model.save(model_filepath)
        print("Model was saved to the path")





        print("----- TrainModel --> train_with_VoxNet  finished!!")
