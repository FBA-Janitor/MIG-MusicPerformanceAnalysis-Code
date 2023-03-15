import argparse

from src.Training import training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--trainingDirectory', type=str,
                            help="training file directory")
    parser.add_argument('-o', '--writeAddress', type=str,
                            help="output model directory")
    parser.add_argument('-n', '--modelFileName', type=str,
                            help="output model file name")
    parser.add_argument('--isAudio', action='store_true',
                            help="whether the training file is audio")
    parser.add_argument('-gt', '--textAddress', type=str, required=False, default='',
                            help="annotated file directory")
    args = parser.parse_args()
    
    training(
        args.trainingDirectory,
        args.writeAddress,
        args.modelFileName,
        args.isAudio,
        args.textAddress)