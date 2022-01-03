from Thesis_Lib import *
from playsound import playsound

# -----------------  Dataset Load  START ---------------------- #
# obj_Dataset = Dataset(prefixDir= "./Datasets/ModelNet40/",
#                       prefixDirOutput= "./Datasets/ModelNet40_output/")
# obj_Dataset.deleteBinvoxFiles()
# obj_Dataset.create_binvoxFiles_from_offFiles(shapeNumber= 16)
# obj_Dataset.create_numpyFiles_from_binvoxFiles(shapeNumber= 16)
# playsound("./Sounds/a_10_Guage_Shotgun.wav")
# del obj_Dataset
# -----------------  Dataset Load  END   ---------------------- #



# -----------------  Train Model  START  ---------------------- #
obj_TrainOp = TrainOP(
                      # prefixDirOutput= "./Datasets/ModelNet10_output/",
                      # filename__x_train="./Datasets/ModelNet10_output/x_train_shape32_1_1_2022__23_26_38.npy",
                      # filename__y_train= "./Datasets/ModelNet10_output/y_train_shape32_1_1_2022__23_26_38.npy",
                      prefixDirOutput= "./Datasets/ModelNet40_output/",
                      filename__x_train="./Datasets/ModelNet40_output/x_train_shape32_1_1_2022__23_32_54.npy",
                      filename__y_train="./Datasets/ModelNet40_output/y_train_shape32_1_1_2022__23_32_54.npy"
                      )
obj_TrainOp.build_and_train_ModelNet()

playsound("./Sounds/a_10_Guage_Shotgun.wav")
del obj_TrainOp

# obj_TrainModel.train_with_VoxNet(x_train= x_train, x_valid= x_valid,
#                                  y_train= y_train, y_valid= y_valid)
# playsound("./Sounds/a_10_Guage_Shotgun.wav")
# -----------------  Train Model  END    ---------------------- #


# voxnet bildirisinde yazdığı gibi train et __ 2 sa
# yeni dataset bul ve voxnet ile dene  __ 2 sa


# training detaylarını araştır
# # - initializer çeşitleri
# # - optimizer çeşitleri
# # - loss ve metric çitleri
# voxelization kullanan tüm makaleleri bul
# depth kamera ile görüntü elde et


