import base64
import logging
import os
from collections import OrderedDict
from io import BytesIO

import cv2
import numpy as np
import torch

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
    r = torch.ones([M.shape[0], M.shape[1] - 1],device='cpu')
    r = torch.cat([r, torch.ones([M.shape[0], 1],device='cpu') * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1],device='cpu')
    c = torch.cat([c, torch.ones([M.shape[0], 1],device='cpu') * M.shape[2]], dim=-1)
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

        mask_mc = index2[index] == torch.arange(len(p)).to(self.device)
        mask = mask_th & mask_mc
        index1, index2 = torch.nonzero(mask).squeeze(1), index[mask]
        final_score = p[index1, index2]
        return index1, index2, final_score


    def initialize(self, ctx):
        """
        load eager mode state_dict based model
        """

        properties = ctx.system_properties
        self.device = torch.device(
            "cpu"
        )

        logger.info(f"Device on initialization is: {self.device}")

        manifest = ctx.manifest
        logger.error(manifest)

        model_dir = properties.get("model_dir")
        serialized_file = manifest["model"]["serializedFile"]

        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model definition file")

        from model import SG_Model

        self.model = SG_Model(self.config)
        self.model.to(self.device), self.model.eval()
        checkpoint = torch.load(model_pt_path, map_location=self.device)

        # load the state dict
        if list(checkpoint['state_dict'].items())[0][0].split('.')[0] == 'module':
            new_stat_dict = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                new_stat_dict[key[7:]] = value
            checkpoint['state_dict'] = new_stat_dict
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        # load image

        logger.info("len of data[0] is %d "%len(data[0]))
        logger.info(data[0].keys())
        logger.info(len(data))
        logger.info("len of values %d " % len(data[0].values()))
        logger.info("len of image1  %d " % len(data[0]["file1"]))
        logger.info("type of image1  %s " % str(type(data[0]["file1"])))

        # read image
        img1_bype = data[0].get("file1")
        logger.info(img1_bype)
        if img1_bype is None:
            img1_bype = data[0].get("body")
            logging.info(data[0].get("body"))
        logger.info("img1_bype.shape %s " % str(len(img1_bype)))
        logger.info(img1_bype)
        nparr = np.frombuffer(img1_bype, np.uint8)
        logger.info("nparr.shape %s " % str(nparr.shape))
        img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info("img1.shape %s " % str(img1.shape))
        logger.info("img1.mean %s " % str(np.mean(img1)))

        img2_bype = data[0].get("file2")
        if img2_bype is None:
            img2_bype = data[1].get("body")

        nparr = np.frombuffer(img2_bype, np.uint8)
        logger.info("typre of nparr %s" % str(type(nparr)))

        img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info("img1.shape %s " % str(img2.shape))
        logger.info("img2.mean %s " % str(np.mean(img2)))


        kpt1_byte = data[0].get("kpt1")
        if kpt1_byte is None:
            kpt1_byte = data[2].get("body")

        kpt1_buffer = BytesIO(kpt1_byte)
        kpt1 = np.load(kpt1_buffer)
        logger.info("kpt1 1 shape %s " % (str(kpt1.shape)))
        logger.info("kpt1.mean %s " % str(np.mean(kpt1)))


        desc1_byte = data[0].get("desc1")
        if desc1_byte is None:
            desc1_byte = data[3].get("body")

        desc1_buffer = BytesIO(desc1_byte)
        desc1 = np.load(desc1_buffer)
        logger.info("desc1 1 shape %s " % (str(desc1.shape)))
        logger.info("desc1.mean %s " % str(np.mean(desc1)))


        kpt2_byte = data[0].get("kpt2")
        if kpt2_byte is None:
            kpt2_byte = data[4].get("body")

        kpt2_buffer = BytesIO(kpt2_byte)
        kpt2 = np.load(kpt2_buffer)
        logger.info("kpt2 2 shape %s " % (str(kpt2.shape)))
        logger.info("kpt2.mean %s " % str(np.mean(kpt2)))


        desc2_byte = data[0].get("desc2")
        if desc2_byte is None:
            desc2_byte = data[5].get("body")

        desc2_buffer = BytesIO(desc2_byte)
        desc2 = np.load(desc2_buffer)
        logger.info("desc 2 shape %s " % (str(desc2.shape)))
        logger.info("desc2.mean %s " % str(np.mean(desc2)))

        """
        # read image with PIL
        # img1 = Image.open(io.BytesIO(img1_bype)).convert('RGB')
        # logger.info("img1 PIL by %s " % str(img1.size))
        """
        size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(np.asarray(img2.shape[:2]))
        logger.info("size 1 size2 %s %s" % (size1, size2))

        # compose a new dictionary
        test_data = {'x1': kpt1, 'x2': kpt2, 'desc1': desc1, 'desc2': desc2, 'size1': size1, 'size2': size2}
        self.data = test_data


        return test_data

    def inference(self, data):

        norm_x1, norm_x2 = normalize_size(data['x1'][:, :2], data['size1']), \
                           normalize_size(data['x2'][:, :2], data['size2'])
        x1, x2 = np.concatenate([norm_x1, data['x1'][:, 2, np.newaxis]], axis=-1), np.concatenate(
            [norm_x2, data['x2'][:, 2, np.newaxis]], axis=-1)
        feed_data = {'x1': torch.from_numpy(x1[np.newaxis]).to(self.device).float(),
                     'x2': torch.from_numpy(x2[np.newaxis]).to(self.device).float(),
                     'desc1': torch.from_numpy(data['desc1'][np.newaxis]).to(self.device).float(),
                     'desc2': torch.from_numpy(data['desc2'][np.newaxis]).to(self.device).float()}
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

        encoded_dictionary = bytes(str(matching_result), "utf-8")
        test = [
            {
                "base64_prediction": base64.b64encode(
                    encoded_dictionary
                ).decode("utf-8")
            }
        ]

        logger.info(test)
        return test


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
torch-model-archiver --model-name superglue --version 1.0 --model-file deployment/model.py --serialized-file weights/model_best.pth --export-path model_store --handler deployment/handler.py -f

torchserve --start --model-store model_store --models model_store/superglue.mar --ts-config deployment/config.properties

send request
curl -H "Content-Type: application/json" -X POST "http://localhost:8080/predictions/superglue/1.0" -d '{"image1_url": "@demo/demo_1.jpg","image2_url": "@demo/demo_1.jpg","img1_kpt1": "./demo/demo1_kpt.npy","img1_desc1": "./demo/demo1_desc.npy","img1_kpt2": "./demo/demo1_kpt.npy","img1_desc2": "./demo/demo1_desc.npy"}'

time curl -X POST http://localhost:8080/predictions/superglue/1.0 -F 'file1=@demo/demo_1.jpg' -F 'file2=@demo/demo_2.jpg' -F 'kpt1=@demo/demo1_kpt.npy' -F 'desc1=@demo/demo1_desc.npy' -F 'kpt2=@demo/demo2_kpt.npy' -F 'desc2=@demo/demo2_desc.npy'
curl http://localhost:8080/predictions/superglue/1.0 -F 'file1=@demo/demo_1.jpg' -F 'file2=@demo/demo_2.jpg' -F 'kpt1=@demo/demo1_kpt.npy' -F 'desc1=@demo/demo1_desc.npy' -F 'kpt2=@demo/demo2_kpt.npy' -F 'desc2=@demo/demo2_desc.npy'
curl http://localhost:8080/predictions/superglue/1.0 -T 'file1=@demo/demo_1.jpg' -T 'file2=@demo/demo_2.jpg' -T 'kpt1=@demo/demo1_kpt.npy' -T 'desc1=@demo/demo1_desc.npy' -T 'kpt2=@demo/demo2_kpt.npy' -T 'desc2=@demo/demo2_desc.npy'
curl -X PUT --data-binary 'file1=@demo/demo_1.jpg' --data-binary 'file2=@demo/demo_2.jpg' --data-binary 'kpt1=@demo/demo1_kpt.npy' --data-binary 'desc1=@demo/demo1_desc.npy' --data-binary 'kpt2=@demo/demo2_kpt.npy' --data-binary 'desc2=@demo/demo2_desc.npy' http://localhost:8080/predictions/superglue/1.0
curl -X PUT --data-urlencode file1@demo/demo_1.jpg --data-urlencode file2@demo/demo_2.jpg --data-urlencode kpt1@demo/demo1_kpt.npy --data-urlencode desc1@demo/demo1_desc.npy --data-urlencode kpt2@demo/demo2_kpt.npy --data-urlencode desc2@demo/demo2_desc.npy http://localhost:8080/predictions/superglue/1.0

curl -X PUT --data-binary file1=@demo/demo_1.jpg http://localhost:8080/predictions/superglue/1.0
--data-urlencode file2@demo/demo_2.jpg --data-urlencode kpt1@demo/demo1_kpt.npy http://localhost:8080/predictions/superglue/1.0


curl -F "file1=@demo/demo_1.jpg" -F 'file2=@demo/demo_2.jpg' -F 'kpt1=@demo/demo1_kpt.npy' -F 'desc1=@demo/demo1_desc.npy' -F 'kpt2=@demo/demo2_kpt.npy' -F 'desc2=@demo/demo2_desc.npy' http://localhost:8080/predictions/superglue/1.0
"""