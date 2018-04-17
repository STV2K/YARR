
# Paths
__sample_path_1__ = "../testcases/no_anno/SP_LOFI_mmexport1519399281474_WX.jpg"
__sample_path_2__ = "../testcases/STV2K_ts_0339.jpg"

training_data_path = "/Users/dementia/Desktop/Research/Datasets/St2kNew/stv2k_train"
test_data_path = "/Users/dementia/Desktop/Research/Datasets/St2kNew/stv2k_test"
training_data_path_z440 = "/home/xhuang/Research/Datasets/STV2K_New/stv2k_train"
demo_data_path = "../testcases/"

# Constants
__STV2KImageWidth = 2448
__STV2KImageHeight = 3264

# Data and Augmentation Settings
min_crop_side_ratio = 0.2
min_text_size = 7
min_char_avgsize = 25
max_side_len = 1280

# Detection Branch Settings
text_scale = 1024  # Decides the receptive field of detection branch
score_map_threshold = 0.8
box_threshold = 0.1
nms_threshold = 0.2



