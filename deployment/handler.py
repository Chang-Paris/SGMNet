import os
import torch
import logging
logger = logging.getLogger(__name__)


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
            "sink_iter": [10,100],
            "head": 4,
            "net_channels": 256
        })  # Model definition configuration => todo clean this


    def initialize(self, ctx):
        """
        load eager mode state_dict based model
        """

        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Device on initialization is: {self.device}")
        model_dir = properties.get("model_dir")

        manifest = ctx.manifest
        logger.error(manifest)
        serialized_file = manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model definition file")

        from model import SG_Model
        self.model = SG_Model(self.config)
        self.model.cuda(), self.model.eval()
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # todo load the state dict

    def preprocess(self, data):
        info_pair = dict()
        return info_pair

    def inference(self, info_pair):
        prediction = dict()
        return prediction

    def postprocess(self, prediction):
        matching_result = dict()
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
