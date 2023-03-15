import os
import argparse

from src.ProcessInput import writeFeatureData
from src.Testing_Evaluation import testing

def evaluate(testingDirectory, modelPath, outputDirectory, groundTruthDirectory):
    tmp_feature_dir = os.path.join(testingDirectory, 'tmp')
    if not os.path.exists(tmp_feature_dir):
        os.mkdir(tmp_feature_dir)
    print("Extracting features ...")
    writeFeatureData(testingDirectory, groundTruthDirectory, tmp_feature_dir, None, True, False)

    testing(tmp_feature_dir, modelPath, outputDirectory)

    for file in os.listdir(tmp_feature_dir):
        os.remove(os.path.join(tmp_feature_dir, file))
    os.rmdir(tmp_feature_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--testingDirectory', type=str,
                            help="testing file directory")
    parser.add_argument('-o', '--outputDirectory', type=str,
                            help="output txt report directory")
    parser.add_argument('-m', '--modelPath', type=str,
                            help="path of saved model file")
    parser.add_argument('-gt', '--groundTruth', type=str,
                            help="annotated file directory")
    args = parser.parse_args()

    evaluate(
        args.testingDirectory,
        args.modelPath,
        args.outputDirectory,
        args.groundTruth
    )

    # testing(
    #     args.testingDirectory,
    #     args.modelPath,
    #     args.outputDirectory
    # )