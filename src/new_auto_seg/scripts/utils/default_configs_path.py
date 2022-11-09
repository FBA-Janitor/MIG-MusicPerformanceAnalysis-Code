audio_dir = "/home/yding402/fba-data/MIG-FBA-Data-Cleaning/cleaned/audio"
segment_dir = "/home/yding402/fba-data/MIG-FBA-Data-Cleaning/cleaned/segmentation"

# For general use
feature_write_dir = "../tmp_feature"
norm_param_path = "utils/norm_params"

# For generating (student id, audio filepath, segment filepath) csv file
model_load_path = "../model_save/2022test.pth"
segment_status_csv = "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/data_parse/config/segmentation/segment_status_230922.csv"

evaluate_csv_list = [
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_AltoSaxophone_audio_seg.csv",
    # "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_BbClarinet_audio_seg.csv"
]

# For training
model_save_path = "../model_save/2022test.pth"
train_csv_list = [
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_AltoSaxophone_audio_seg.csv",
    # "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_BbClarinet_audio_seg.csv"
]