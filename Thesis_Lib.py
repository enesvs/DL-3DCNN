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

    def __init__(self, prefixDir, prefixDirOutput):
        print("\n----- Dataset Initialization  started!! -----")
        time.sleep(2)

        self.prefixDir = prefixDir
        self.prefixDirOutput = prefixDirOutput

        self.classesInfo = {"listOfClasses": list(),
                            "numberOfSample_in_trainFolder": dict(),
                            "minNumberOfSample_in_trainFolder": 0,
                            "BINVOXsampleNameList_in_trainFolder": dict(),
                            "OFFsampleNameList_in_trainFolder" : dict(),

                            "numberOfSample_in_testFolder": dict(),
                            "minNumberOfSample_in_testFolder": 0,
                            "BINVOXsampleNameList_in_testFolder": dict(),
                            "OFFsampleNameList_in_testFolder": dict(),
                            }

        for file in os.listdir(self.prefixDir):
            if os.path.isdir(os.path.join(self.prefixDir, file)):
                self.classesInfo["listOfClasses"].append(file)

        self.classesInfo["listOfClasses"].sort()



        for classname in self.classesInfo["listOfClasses"]:
            temp = list()
            temp2 = list()
            for fileName in os.listdir(self.prefixDir + classname + "/train"):
                if ".off" in fileName:
                    temp.append(fileName)
                if ".binvox" in fileName:
                    temp2.append(fileName)

            temp.sort()
            temp2.sort()
            self.classesInfo["OFFsampleNameList_in_trainFolder"][classname] = temp
            self.classesInfo["BINVOXsampleNameList_in_trainFolder"][classname] = temp2
            self.classesInfo["numberOfSample_in_trainFolder"][classname] = len(temp)
            # print("asdasd ", self.classesInfo["OFFsampleNameList_in_trainFolder"][classname])



        for classname in self.classesInfo["listOfClasses"]:
            temp = list()
            temp2 = list()
            for fileName in os.listdir(self.prefixDir + classname + "/test"):
                if ".off" in fileName:
                    temp.append(fileName)
                if ".binvox" in fileName:
                    temp2.append(fileName)

            temp.sort()
            temp2.sort()
            self.classesInfo["OFFsampleNameList_in_testFolder"][classname] = temp
            self.classesInfo["BINVOXsampleNameList_in_testFolder"][classname] = temp2
            self.classesInfo["numberOfSample_in_testFolder"][classname] = len(temp)

        self.classesInfo["minNumberOfSample_in_trainFolder"] = min(self.classesInfo["numberOfSample_in_trainFolder"].values())
        self.classesInfo["minNumberOfSample_in_testFolder"] = min(self.classesInfo["numberOfSample_in_testFolder"].values())

        firstClassName_in_trainFolder = self.classesInfo["listOfClasses"][0]

        print("listOfClasses:", self.classesInfo["listOfClasses"],
              "\nminNumberOfSample_in_trainFolder", self.classesInfo["minNumberOfSample_in_trainFolder"],
              "\nminNumberOfSample_in_testFolder", self.classesInfo["minNumberOfSample_in_testFolder"],
              "\nOFFsampleNameList_in_trainFolder", self.classesInfo["OFFsampleNameList_in_trainFolder"][ classname ][0:10],
              "\nOFFsampleNameList_in_testFolder", self.classesInfo["OFFsampleNameList_in_testFolder"][ classname ][0:10])

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
            for sampleName in self.classesInfo["OFFsampleNameList_in_trainFolder"][cl_name][0:minNumber_train]:
                os.system(
                    "./binvox -c -d " + str(
                        shapeNumber) + " -e -t binvox -cb " + self.prefixDir + classname + "/train" + "/" + sampleName)
                totalCounter = totalCounter + 1

        minNumber_test = self.classesInfo["minNumberOfSample_in_testFolder"]
        for classname in self.classesInfo["listOfClasses"]:
            cl_name = classname
            for sampleName in self.classesInfo["OFFsampleNameList_in_testFolder"][cl_name][0:minNumber_test]:

                os.system(
                    "./binvox -c -d " + str(
                        shapeNumber) + " -e -t binvox -cb " + self.prefixDir + classname + "/test" + "/" + sampleName)
                totalCounter = totalCounter + 1

        print("{} files were created.".format(totalCounter))
        print("----- create_binvoxFiles_from_offFiles  finished!!")

    def create_numpyFiles_from_binvoxFiles(self, shapeNumber):
        print("----- create_numpyFiles_from_binvoxFiles  started!!")

        passOrbreak = 0
        for classname in self.classesInfo["listOfClasses"]:
            if self.classesInfo["minNumberOfSample_in_trainFolder"] != len(self.classesInfo["BINVOXsampleNameList_in_trainFolder"][classname]):
                passOrbreak = 1
                print("Eksik bin dosyası var!!",
                      "\nminNumberOfSample_in_trainFolder:", self.classesInfo["minNumberOfSample_in_trainFolder"],
                      "\t\tlen(OFFsampleNameList_in_trainFolder in {}): ".format(classname), len(self.classesInfo["BINVOXsampleNameList_in_trainFolder"][classname]) )

            if self.classesInfo["minNumberOfSample_in_testFolder"] != len(self.classesInfo["BINVOXsampleNameList_in_testFolder"][classname]):
                passOrbreak = 1
                print("Eksik bin dosyası var!!",
                      "\nminNumberOfSample_in_testFolder:", self.classesInfo["minNumberOfSample_in_testFolder"],
                      "\t\tlen(OFFsampleNameList_in_testFolder in {}): ".format(classname), len(self.classesInfo["BINVOXsampleNameList_in_testFolder"][classname]) )

        if passOrbreak == 1:
            exit()
        else:
            pass


        totalNUmberOfSamples = self.classesInfo["minNumberOfSample_in_trainFolder"] * len(self.classesInfo["listOfClasses"])
        x_train = np.zeros(shape=(totalNUmberOfSamples, shapeNumber, shapeNumber, shapeNumber), dtype=float)
        y_train = np.zeros((totalNUmberOfSamples,), dtype="int")

        totalNUmberOfSamples2 = self.classesInfo["minNumberOfSample_in_testFolder"] * len(self.classesInfo["listOfClasses"])
        x_test = np.zeros(shape=(totalNUmberOfSamples2, shapeNumber, shapeNumber, shapeNumber), dtype=float)
        y_test = np.zeros((totalNUmberOfSamples2,), dtype="int")


        currentNumberOfSample = 0
        currentNumberOfSample2 = 0
        for label, classname in enumerate(self.classesInfo["listOfClasses"]):
            for binvoxFileName in self.classesInfo["BINVOXsampleNameList_in_trainFolder"][classname]:
                with open(self.prefixDir + classname + "/train/" + binvoxFileName, 'rb') as f:
                    model = binvox_rw.read_as_3d_array(f)
                    narray = np.ndarray(shape=(1, shapeNumber, shapeNumber, shapeNumber), dtype=float)
                    narray = model.data
                    x_train[currentNumberOfSample, :] = narray.astype(float)
                    y_train[currentNumberOfSample] = label
                    currentNumberOfSample = currentNumberOfSample +1
                    # print("currentNumberOfSample: ", currentNumberOfSample)

            for binvoxFileName2 in self.classesInfo["BINVOXsampleNameList_in_testFolder"][classname]:
                with open(self.prefixDir + classname + "/test/" + binvoxFileName2, 'rb') as f:
                    model2 = binvox_rw.read_as_3d_array(f)
                    narray2 = np.ndarray(shape=(1, shapeNumber, shapeNumber, shapeNumber), dtype=float)
                    narray2 = model2.data
                    x_test[currentNumberOfSample2, :] = narray2.astype(float)
                    y_test[currentNumberOfSample2] = label
                    currentNumberOfSample2 = currentNumberOfSample2 +1
                    # print("currentNumberOfSample2: ", currentNumberOfSample2)

        dt = datetime.datetime.today()

        print("x_train.shape", x_train.shape)
        np.save(self.prefixDirOutput + "x_train_shape{}_{}_{}_{}__{}_{}_{}.npy".format(shapeNumber, dt.day,
                                                                                       dt.month, dt.year,
                                                                                       dt.hour, dt.minute,
                                                                                       dt.second), x_train)
        y_train_categorical = to_categorical(y_train, len(self.classesInfo["listOfClasses"]))
        print("y_train_categorical.shape", y_train_categorical.shape)
        np.save(self.prefixDirOutput + "y_train_shape{}_{}_{}_{}__{}_{}_{}.npy".format(shapeNumber, dt.day,
                                                                                      dt.month, dt.year,
                                                                                      dt.hour, dt.minute,
                                                                                      dt.second), y_train_categorical)


        print("x_test.shape", x_test.shape)
        np.save(self.prefixDirOutput + "x_test_shape{}_{}_{}_{}__{}_{}_{}.npy".format(shapeNumber, dt.day,
                                                                                       dt.month, dt.year,
                                                                                       dt.hour, dt.minute,
                                                                                       dt.second), x_test)

        y_test_categorical = to_categorical(y_test, len(self.classesInfo["listOfClasses"]))
        print("y_test_categorical.shape", y_test_categorical.shape)
        np.save(self.prefixDirOutput + "y_test_shape{}_{}_{}_{}__{}_{}_{}.npy".format(shapeNumber, dt.day,
                                                                                       dt.month, dt.year,
                                                                                       dt.hour, dt.minute,
                                                                                       dt.second), y_test_categorical)

        print("----- create_numpyFiles_from_binvoxFiles  finished!!")


class TrainOP:

    def __init__(self, prefixDirOutput, filename__x_train, filename__y_train):
        print("\n----- TrainOP Initialization  started!! -----")
        self.prefixDirOutput = prefixDirOutput

        dt = datetime.datetime.today()
        self.prefixForOutputFileName = "output_{}{}{}__{}_{}_{}__".format(dt.day, dt.month, dt.year,
                                                                         dt.hour, dt.minute, dt.second)

        y_train_total = np.load(filename__y_train)
        # y_train_total = y_train_total.astype(float)
        x_train_total = np.load(filename__x_train)
        x_train_total = x_train_total.astype(float)
        x_train_total = np.expand_dims(x_train_total, axis=4)
        print("x_train_total shape: ", x_train_total.shape, "\ty_train_total shape: ", y_train_total.shape)

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_train_total, y_train_total, test_size=0.2, random_state=1)
        print("\nx_train shape: ", self.x_train.shape, "\ty_train shape: ", self.y_train.shape)
        print("x_valid shape: ", self.x_valid.shape, "\ty_valid shape: ", self.y_valid.shape)

        print("\n----- TrainOP Initialization  finished!! -----")

    def build_and_train_ModelNet(self):
        print("\n----- build_and_train  started!! -----")

        #### Nested definiton START ####
        def my_initializer(shape, dtype=None):
            receptive_field_size = np.prod(shape[2:])
            c = shape[1]  # input channels
            nl = c * receptive_field_size
            std = np.sqrt(2.0 / (nl))
            return np.random.normal(0, std, size=shape).astype('float32')

        #### Nested definiton END   ####

        input_shape = self.x_train.shape[1:5]
        num_classes = self.y_train.shape[1]
        print("input shape: ", input_shape, "num classes: ", num_classes)

        inputs1 = Input(shape=input_shape)

        layer1 = Conv3D(filters=32, kernel_size=5, strides=(2, 2, 2), padding="valid",
                        activation='relu', kernel_initializer=my_initializer)(inputs1)
        layer1 = Dropout(0.2)(layer1)

        layer2 = Conv3D(filters=32, kernel_size=3, strides=(1, 1, 1), padding="valid",
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

        filepath = self.prefixDirOutput + self.prefixForOutputFileName + "Architecture.png"
        ## ""sudo apt-get install graphviz""
        ## terminal'e bunu yazmalısın. virEnv bağımsız yüklemelisin.
        plot_model(model, show_shapes=True, to_file=filepath)

        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(x=self.x_train, y=self.y_train, validation_data=(self.x_valid, self.y_valid),
                            epochs=10, batch_size=16, verbose=2)

        model_filepath = self.prefixDirOutput + self.prefixForOutputFileName + "Model.h5"
        model.save(model_filepath)
        print("Model was saved to the path")

        print("\n----- build_and_train  finished!! -----")




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
