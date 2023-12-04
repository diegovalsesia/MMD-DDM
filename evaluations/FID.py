
import os
import torch
import argparse

from pytorch_fid.fid_score import calculate_fid_given_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    kwargs = {'batch_size': 50, 'device': torch.device('cuda:0'), 'dims': 2048}
    paths = ['/celeba_fid_statistics.npz','./celeb_test/statistics.npz']
    fid = calculate_fid_given_paths(paths=paths, **kwargs)
    print(' FID = {}'.format(fid))