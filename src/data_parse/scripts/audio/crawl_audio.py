import urllib.request

import pandas as pd



def download_one(args):
    src = args.src
    dst = args.dst

    urllib.request.urlretrieve(src, dst)


def process_xls(xls_path):
    
    df = pd.read_excel(xls_path)