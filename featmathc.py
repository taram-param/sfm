import cv2
import numpy as np
import pickle
import argparse
import os
from time import time

from utils import *


class Options:

    def __init__(self):

        self.data_files = []
        self.data_dir = "data/fountain-P11/images/"
        self.ext = ["jpg", "png"]
        self.out_dir = "data/fountain-P11/"

        self.features = "SIFT"
        self.matcher = "BFMatcher"
        self.cross_check = True

        # [1,+inf] print progress every print_every seconds, -1 to disable (default: 1)
        self.print_every = 1
        # [True|False] whether to save images with keypoints drawn on them (default: False)
        self.save_results = False


def FeatMatch(opts, data_files=[]):
    if len(data_files) == 0:
        img_names = sorted(os.listdir(opts.data_dir))
        img_paths = [os.path.join(opts.data_dir, x) for x in img_names if \
                     x.split('.')[-1] in opts.ext]

    else:
        img_paths = data_files
        img_names = sorted([x.split('/')[-1] for x in data_files])

    feat_out_dir = os.path.join(opts.out_dir, 'features', opts.features)
    matches_out_dir = os.path.join(opts.out_dir, 'matches', opts.matcher)

    if not os.path.exists(feat_out_dir):
        os.makedirs(feat_out_dir)
    if not os.path.exists(matches_out_dir):
        os.makedirs(matches_out_dir)

    data = []
    t1 = time()
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img_name = img_names[i].split('.')[0]
        img = img[:, :, ::-1]

        feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
        kp, desc = feat.detectAndCompute(img, None)
        data.append((img_name, kp, desc))

        kp_ = SerializeKeypoints(kp)

        with open(os.path.join(feat_out_dir, 'kp_{}.pkl'.format(img_name)), 'wb') as out:
            pickle.dump(kp_, out)

        with open(os.path.join(feat_out_dir, 'desc_{}.pkl'.format(img_name)), 'wb') as out:
            pickle.dump(desc, out)

        if opts.save_results:
            raise NotImplementedError

        t2 = time()

        if (i % opts.print_every) == 0:
            print('FEATURES DONE: {0}/{1} [time={2:.2f}s]'.format(i + 1, len(img_paths), t2 - t1))

        t1 = time()

    num_done = 0
    num_matches = ((len(img_paths) - 1) * (len(img_paths))) / 2

    t1 = time()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            img_name1, kp1, desc1 = data[i]
            img_name2, kp2, desc2 = data[j]

            matcher = getattr(cv2, opts.matcher)(crossCheck=opts.cross_check)
            matches = matcher.match(desc1, desc2)

            matches = sorted(matches, key=lambda x: x.distance)
            matches_ = SerializeMatches(matches)

            pickle_path = os.path.join(matches_out_dir, 'match_{}_{}.pkl'.format(img_name1,
                                                                                 img_name2))
            with open(pickle_path, 'wb') as out:
                pickle.dump(matches_, out)

            num_done += 1
            t2 = time()

            if (num_done % opts.print_every) == 0:
                print('MATCHES DONE: {0}/{1} [time={2:.2f}s]'.format(num_done, num_matches, t2 - t1))

            t1 = time()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    opts = Options()

    FeatMatch(opts, opts.data_files)