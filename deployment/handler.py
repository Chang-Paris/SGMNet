import os
import logging
import cv2
import torch
import numpy as np

from collections import OrderedDict

logger = logging.getLogger(__name__)


eps=1e-8

def sinkhorn(M,r,c,iteration):
    p = torch.softmax(M, dim=-1)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iteration):
        u = r / ((p * v.unsqueeze(-2)).sum(-1) + eps)
        v = c / ((p * u.unsqueeze(-1)).sum(-2) + eps)
    p = p * u.unsqueeze(-1) * v.unsqueeze(-2)
    return p

def sink_algorithm(M,dustbin,iteration):
    M = torch.cat([M, dustbin.expand([M.shape[0], M.shape[1], 1])], dim=-1)
    M = torch.cat([M, dustbin.expand([M.shape[0], 1, M.shape[2]])], dim=-2)
    r = torch.ones([M.shape[0], M.shape[1] - 1],device='cuda')
    r = torch.cat([r, torch.ones([M.shape[0], 1],device='cuda') * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1],device='cuda')
    c = torch.cat([c, torch.ones([M.shape[0], 1],device='cuda') * M.shape[2]], dim=-1)
    p=sinkhorn(M,r,c,iteration)
    return p

def normalize_size(x,size,scale=1):
    size=size.reshape([1,2])
    norm_fac=size.max()
    return (x-size/2+0.5)/(norm_fac*scale)


class SuperGlueHandler:
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.p_th = 0.2  # matching score threshold
        self.config = dict({
            "use_score_encoding": True,
            "layer_num": 9,
            "sink_iter": [10, 100],
            "head": 4,
            "net_channels": 256
        })  # Model definition configuration => todo clean this
        self.data = None

    def match_p(self, p):  # p N*M
        # return the score of matching
        score, index = torch.topk(p, k=1, dim=-1)  # take max score by row
        _, index2 = torch.topk(p, k=1, dim=-2)  # take max score by column

        mask_th, index, index2 = score[:, 0] > self.p_th, index[:, 0], index2.squeeze(0)  # filter by score threshold

        mask_mc = index2[index] == torch.arange(len(p)).cuda()
        mask = mask_th & mask_mc
        index1, index2 = torch.nonzero(mask).squeeze(1), index[mask]
        final_score = p[index1, index2]
        return index1, index2, final_score

    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def initialize(self, ctx):
        """
        load eager mode state_dict based model
        """

        self.device = self.get_device()

        logger.info(f"Device on initialization is: {self.device}")
        model_dir = ctx.get("model_dir")

        serialized_file = ctx["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model definition file")

        from deployment.model import SG_Model

        self.model = SG_Model(self.config)
        self.model.cuda(), self.model.eval()
        checkpoint = torch.load(model_pt_path, map_location=self.device)

        # load the state dict
        if list(checkpoint['state_dict'].items())[0][0].split('.')[0] == 'module':
            new_stat_dict = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                new_stat_dict[key[7:]] = value
            checkpoint['state_dict'] = new_stat_dict
        self.model.load_state_dict(checkpoint['state_dict'])

    def preprocess(self, data):
        # load image
        img1, img2 = cv2.imread(data["image1_url"]), cv2.imread(data["image2_url"])
        size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(np.asarray(img2.shape[:2]))

        # load the input kp and descriptors
        kpt1 = np.load(data["img1_kpt1"])
        desc1 = np.load(data["img1_desc1"])
        kpt2 = np.load(data["img1_kpt2"])
        desc2 = np.load(data["img1_desc2"])

        # compose a new dictionary
        test_data = {'x1': kpt1, 'x2': kpt2, 'desc1': desc1, 'desc2': desc2, 'size1': size1, 'size2': size2}
        self.data = test_data
        return test_data

    def inference(self, data):
        norm_x1, norm_x2 = normalize_size(data['x1'][:, :2], data['size1']), \
                           normalize_size(data['x2'][:, :2], data['size2'])
        x1, x2 = np.concatenate([norm_x1, data['x1'][:, 2, np.newaxis]], axis=-1), np.concatenate(
            [norm_x2, data['x2'][:, 2, np.newaxis]], axis=-1)
        feed_data = {'x1': torch.from_numpy(x1[np.newaxis]).cuda().float(),
                     'x2': torch.from_numpy(x2[np.newaxis]).cuda().float(),
                     'desc1': torch.from_numpy(data['desc1'][np.newaxis]).cuda().float(),
                     'desc2': torch.from_numpy(data['desc2'][np.newaxis]).cuda().float()}
        with torch.no_grad():
            res = self.model(feed_data, test_mode=True)
            prediction = res['p']  # matching result matrix => depends on nbr of kp as input
        return prediction

    def postprocess(self, prediction):
        index1, index2, scores = self.match_p(prediction[0, :-1, :-1])
        corr1, corr2 = self.data['x1'][:, :2][index1.cpu()], self.data['x2'][:, :2][index2.cpu()]
        if len(corr1.shape) == 1:
            corr1, corr2 = corr1[np.newaxis], corr2[np.newaxis]
        matching_result = dict({
            "demo1_pt": corr1,
            "demo2_pt": corr2,
            "match_score": scores
        })
        return matching_result


_service = SuperGlueHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data

"""
test field

# define data
data = dict({
    "image1_url": "../demo/demo_1.jpg",
    "image2_url": "../demo/demo_1.jpg",
    "img1_kpt1": "../demo/demo1_kpt.npy",
    "img1_desc1": "../demo/demo1_desc.npy",
    "img1_kpt2": "../demo/demo1_kpt.npy",
    "img1_desc2": "../demo/demo1_desc.npy",
})

# define contexte
contexte = dict({
    "model_dir": "../weights/",
    "serializedFile": "model_best.pth"
})

match = handle(data=data, context=contexte)
print(match)


# zip the model
torch-model-archiver --model-name superglue --version 1.0 --model-file deployment/model.py --serialized-file model_store/model_best.pth --export-path model_store --handler deployment/handler.py -f
torchserve --start --ncs --model-store model_store --models model_store/superglue.mar

"""