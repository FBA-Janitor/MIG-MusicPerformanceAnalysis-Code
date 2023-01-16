audio_dir = "/home/yding402/fba-data/MIG-FBA-Data-Cleaning/cleaned/audio"
segment_dir = "/home/yding402/fba-data/MIG-FBA-Data-Cleaning/cleaned/segmentation"
algo_segment_dir = "/home/yding402/fba-data/MIG-FBA-Segmentation/cleaned/algo-segmentation"

# For general use
feature_write_dir = "../tmp_feature"
norm_param_path = "utils/norm_params"

# For generating (student id, audio filepath, segment filepath) csv file
model_load_path = "../model_save/2023_happy_new_year_model.pth"
segment_status_csv = "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/data_parse/config/segmentation/segment_status_230922.csv"

evaluate_csv_list = [
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_AltoSaxophone_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_BbClarinet_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_Flute_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_Trumpet_audio_seg.csv"
]

# For training
model_save_path = "../model_save/2022test.pth"
train_csv_list = [
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_AltoSaxophone_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_BbClarinet_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_Flute_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_concert_Trumpet_audio_seg.csv",

    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_middle_AltoSaxophone_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_middle_BbClarinet_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_middle_Flute_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_middle_Trumpet_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2013_middle_Oboe_audio_seg.csv",

    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2014_concert_AltoSaxophone_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2014_concert_BbClarinet_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2014_concert_Flute_audio_seg.csv",

    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2014_middle_AltoSaxophone_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2014_middle_BbClarinet_audio_seg.csv",
    "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/new_auto_seg/audio_seg_data/2014_middle_Flute_audio_seg.csv",
]