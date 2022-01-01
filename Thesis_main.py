from Thesis_Lib import *
from playsound import playsound

# -----------------  Dataset Load  START ---------------------- #
###################  For ModelNet40   START #####################
obj_Dataset = Dataset(prefixDir= "./Datasets/ModelNet10/")
# obj_Dataset.deleteBinvoxFiles()
# obj_Dataset.create_binvoxFiles_from_offFiles(shapeNumber= 32)
obj_ModelNet.forModelNet10_create_numpyFiles_from_binvoxFiles(shapeNumber= 64, minNumberOfSamples= minNumberOfSamples) ## 106
# playsound("./Sounds/a_10_Guage_Shotgun.wav")
del obj_Dataset
###################  For ModelNet10   END   #####################

###################  For ModelNet40   START #####################
# obj_ModelNet = ModelNet()
# obj_ModelNet.forModelNet40_deleteBinvoxFiles()
# minNumberOfSamples = obj_ModelNet.forModelNet40_create_binvoxFiles_from_offFiles(shapeNumber= 64)
# obj_ModelNet.forModelNet40_create_numpyFiles_from_binvoxFiles(shapeNumber= 64, minNumberOfSamples= minNumberOfSamples) ## 64
# playsound("./Sounds/a_10_Guage_Shotgun.wav")
# del obj_ModelNet
###################  For ModelNet40   END   #####################
# -----------------  Dataset Load  END   ---------------------- #




# -----------------  Train Model  START  ---------------------- #
# obj_TrainModel = TrainModel(prefixDir= "./Datasets/ModelNet10_output/")
# x_train, x_valid, y_train, y_valid = obj_TrainModel.load_npyFile(filename__x_train= "./Datasets/ModelNet10_output/x_train_shapeNumber64__1_1_2022__12_26_45.npy",
#                                                                  filename__y_train= "./Datasets/ModelNet10_output/y_train_shapeNumber64_1_1_2022__12_26_45.npy")
# obj_TrainModel.train_with_VoxNet(x_train= x_train, x_valid= x_valid,
#                                  y_train= y_train, y_valid= y_valid)
# playsound("./Sounds/a_10_Guage_Shotgun.wav")



# obj_TrainModel = TrainModel(prefixDir= "./Datasets/ModelNet40_output/")
# x_train, x_valid, y_train, y_valid = obj_TrainModel.load_npyFile(filename__x_train= "./Datasets/ModelNet40_output/x_train_112022__10_51_17.npy",
#                                                                  filename__y_train= "./Datasets/ModelNet40_output/y_train_112022__10_51_17.npy")
# obj_TrainModel.train_with_VoxNet(x_train= x_train, x_valid= x_valid,
#                                  y_train= y_train, y_valid= y_valid)
# playsound("./Sounds/a_10_Guage_Shotgun.wav")

# -----------------  Train Model  END    ---------------------- #