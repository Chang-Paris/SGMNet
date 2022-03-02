import os
import sys

import cv2
import numpy as np
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from components import load_component
from utils import evaluation_utils

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='configs/sgm_config.yaml',
                    help='number of processes.')
parser.add_argument('--img1_path', type=str, default='demo_1.jpg',
                    help='number of processes.')
parser.add_argument('--img2_path', type=str, default='demo_2.jpg',
                    help='number of processes.')

args = parser.parse_args()

if __name__ == '__main__':
    with open(args.config_path, 'r') as f:
        demo_config = yaml.load(f)

    extractor = load_component('extractor', demo_config['extractor']['name'], demo_config['extractor'])

    img1, img2 = cv2.imread(args.img1_path), cv2.imread(args.img2_path)
    size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(np.asarray(img2.shape[:2]))
    kpt1, desc1 = extractor.run(args.img1_path)
    kpt2, desc2 = extractor.run(args.img2_path)
    # todo save kp and descriptor
    with open("./demo1_kpt.npy", "wb+") as f:
        np.save(f, kpt1)
    with open("./demo1_desc.npy", "wb+") as f:
        np.save(f, desc1)
    with open("./demo2_kpt.npy", "wb+") as f:
        np.save(f, kpt2)
    with open("./demo2_desc.npy", "wb+") as f:
        np.save(f, desc2)

    matcher = load_component('matcher', demo_config['matcher']['name'], demo_config['matcher'])
    test_data = {'x1': kpt1, 'x2': kpt2, 'desc1': desc1, 'desc2': desc2, 'size1': size1, 'size2': size2}
    corr1, corr2, scores = matcher.run(test_data)

    match_res = dict({
        "demo1_pt":corr1,
        "demo2_pt":corr2,
        "match_score":scores
    })

    with open("./match_result.pkl", "wb+") as f:
        pickle.dump(match_res, f)

    # draw points
    dis_points_1 = evaluation_utils.draw_points(img1, kpt1)
    dis_points_2 = evaluation_utils.draw_points(img2, kpt2)

    # visualize match
    display = evaluation_utils.draw_match(dis_points_1, dis_points_2, corr1, corr2)
    cv2.imwrite('match.png', display)
