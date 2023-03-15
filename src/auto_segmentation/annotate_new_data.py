import argparse

from src.ProcessInput import writeFeatureData
from src.Classification_Annotation import classifyFeatureData


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--audioDirectory', type=str,
                            help="input audio directory")
    parser.add_argument('-o', '--writeDirectory', type=str,
                            help="output report directory")
    parser.add_argument('-m', '--modelPath', type=str,
                            help="path of saved model file")
    parser.add_argument('-n', '--numberOfMusicalExercises', type=int, default=5,
                            help="numer of exercises in each audio")
    parser.add_argument('--generateDataReport', action='store_false',
                            help="generate data repoert")
    parser.add_argument('--keepNPZFiles', action='store_true',
                            help="keep .npz files")
    
    args = parser.parse_args()

    writeFeatureData(args.audioDirectory, '', args.writeDirectory, [])
    classifyFeatureData(args.writeDirectory, args.writeDirectory, args.modelPath,
        args.generateDataReport, args.keepNPZFiles, args.numberOfMusicalExercises)