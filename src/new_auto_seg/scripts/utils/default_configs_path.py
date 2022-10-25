audio_dir = "/home/yding402/fba-data/MIG-FBA-Data-Cleaning/cleaned/audio"
segment_dir = "/home/yding402/fba-data/MIG-FBA-Data-Cleaning/cleaned/segmentation"

model_load_path = "../model_save/2017ABAI.sav"
model_save_path = "../model_save/2022test.pth"
feature_write_dir = "../tmp_feature"

# For generating (student id, audio filepath, segment filepath) csv file
segment_status_csv = "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/data_parse/config/segmentation/segment_status_230922.csv"

# For training
model_save_path = "../model_save/2022test.pth"
train_csv_list = [
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_AltoSaxophone_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_BbClarinet_audio_seg.csv"
]