
import os
import numpy as np

class prepareDataset:

    # A simple class
    # attribute
    attr1 = "mammal"
    attr2 = "dog"

    # A sample method

    def forModelNet10_deleteBinvoxFiles(self):
        print("----- prepareDataset --> forModelNet10_deleteBinvoxFiles  started!!")

        currentdir = os.getcwd()
        ClassDir_in_ModelNet = currentdir + "/Datasets/" + "ModelNet10/ModelNet10/"
        print("ModelNet içindeki tüm class folder'lar burada olmalı : ", ClassDir_in_ModelNet)

        # klasör isimlerini bul. Onlar ModelNet sınıf isimleridir.
        ListOfClasses = [d for d in os.listdir(ClassDir_in_ModelNet) if
                         os.path.isdir(os.path.join(ClassDir_in_ModelNet, d))]
        # print("\nModelNet10 içindeki sınıf isimlerinin listesi :", ListOfClasses)

        # her sınıfa ait örnek sayılarını listele. minimum örnek sayısı olan sınıfı bul.
        NumberOfSamples_in_theClass = list()
        for classname in ListOfClasses:
            sampleNameList = os.listdir(ClassDir_in_ModelNet + classname + "/train")
            print("classname:", classname, "list: ", sampleNameList)

            for sampleName in sampleNameList:
                if ".binvox" in sampleName:
                    os.remove(ClassDir_in_ModelNet + classname + "/train" + "/" + sampleName)
                    print("'{}' is removing..".format(sampleName))

        print("----- prepareDataset --> forModelNet10_deleteBinvoxFiles  finished!!")

    def forModelNet10(self, shapeNumber):
        self.forModelNet10_deleteBinvoxFiles()

        print("----- prepareDataset --> forModelNet10  started!!")

        currentdir = os.getcwd()
        ClassDir_in_ModelNet = currentdir + "/Datasets/" + "ModelNet10/ModelNet10/"
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



        totalsamplenumber = 0
        i = 0
        label = 0
        dataNumber = minNumberOfSamples * len(ListOfClasses)
        x_train = np.zeros(shape=(dataNumber, shapeNumber, shapeNumber, shapeNumber), dtype=float)
        y_train = np.zeros((dataNumber,), dtype="int")



        for nClass, classname in enumerate(ListOfClasses):
            print("************* {}   {}/{}".format(classname, nClass, len(ListOfClasses)) )
            for nSample in range(minNumberOfSamples):

                number = '{0:04}'.format(nSample)
                # print("*************", classname, "--", number,"         type(number):", type(number))
                # print("*************", "label: ", label)

                # print("converting off file to binvox file -- {}".format(
                #     ClassDir_in_ModelNet + classname + "/train" + "/" + classname + "_" + number + ".off"))
                # !!! write "chmod 755 binvox" in the terminal in the directory which binvox existed in.
                os.system(
                    "./binvox -c -d " + str(
                        shapeNumber) + " -e -t binvox -cb " + ClassDir_in_ModelNet + classname + "/train" + "/" + classname + "_" + number + ".off")

            label = label + 1
        print("*************", classname, "  nClass", nClass)
        print("*************", nSample)
        print("\nMinNumberOfMembers:", minNumberOfSamples)





        #     binvoxfilepath = path + classname + "_" + number + ".binvox"
        #
        #     with open(binvoxfilepath, 'rb') as f:
        #         model = binvox_rw.read_as_3d_array(f)
        #         narray = np.ndarray(shape=(1, shapenumber, shapenumber, shapenumber), dtype=float)
        #         narray = model.data
        #         x_train[i, :] = narray.astype(float)
        #         y_train[i] = label
        #         totalsamplenumber = totalsamplenumber + 1
        #         i = i + 1
        #     # time.sleep(2)
        #     os.remove(binvoxfilepath)
        #
        #
        # label = label + 1

    # np.save(directory + exdir + filename_x, x_train)
    # Y = to_categorical(y_train, len(listOfClasses))
    # print
    # "Y.shape", Y.shape
    # np.save(directory + exdir + filename_y, Y)
    #
    # file = open(directory + exdir + labelnames, "w")
    # for n in range(len(listOfClasses)):
    #     file.write("%s " % (listOfClasses[n]))
    #
    # # file.write(history.history['accuracy'])
    # # file.write("\nValidation Accuracy: ")
    # # file.write(history.history['val_accuracy'])
    # file.close()
    #
    # print
    # "totalsamplenumber: ", totalsamplenumber
    # print
    # "\n***************** off to binvox --- binvox to numpy --- deleting binvox files --- save numpy files"