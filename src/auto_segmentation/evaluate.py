import argparse
from src.Testing_Evaluation import testing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--testingDirectory', type=str,
                            help="testing file directory")
    parser.add_argument('-o', '--txtAddress', type=str,
                            help="output txt report directory")
    parser.add_argument('-m', '--modelPath', type=str,
                            help="path of saved model file")
    args = parser.parse_args()

    testing(
        args.testingDirectory,
        args.modelPath,
        args.txtAddress
    )