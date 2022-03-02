import yaml
import numpy as np
import cv2
import pickle

from components.matchers import GNN_Matcher
from utils import evaluation_utils

if __name__ == '__main__':
    # read config file
    with open("./sg_config.yaml", 'r') as f:
        config = yaml.load(f)

    # build Super Glue model
    model = GNN_Matcher(config)

    # load images
    img1, img2 = cv2.imread("./demo/demo_1.jpg"), cv2.imread("./demo/demo_2.jpg")
    size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(np.asarray(img2.shape[:2]))

    # load the input kp and descriptors
    kpt1 = np.load("./demo/demo1_kpt.npy")
    desc1 = np.load("./demo/demo1_desc.npy")
    kpt2 = np.load("./demo/demo2_kpt.npy")
    desc2 = np.load("./demo/demo2_desc.npy")

    # run prediction
    test_data = {'x1': kpt1, 'x2': kpt2, 'desc1': desc1, 'desc2': desc2, 'size1': size1, 'size2': size2}
    corr1, corr2, scores = model.run(test_data)

    match_res = dict({
        "demo1_pt":corr1,
        "demo2_pt":corr2,
        "match_score":scores
    })

    dis_points_1 = evaluation_utils.draw_points(img1, kpt1)
    dis_points_2 = evaluation_utils.draw_points(img2, kpt2)
    # visualize match
    display = evaluation_utils.draw_match(dis_points_1, dis_points_2, corr1, corr2)
    cv2.imwrite('./result/match.png', display)


    # save the result
    with open("./result/match_result.pkl", "wb+") as f:
        pickle.dump(match_res, f)
