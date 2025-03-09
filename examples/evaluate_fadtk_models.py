from fadtk.model_loader import *
import torch.nn as nn
import os

from timbremetrics.metrics import TimbreMetric

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

models = [
    CLAPLaionModel('audio'), 
    CLAPLaionModel('music'),
    VGGishModel(), 
    MERTModel(layer=6),
    EncodecEmbModel('24k'), 
    EncodecEmbModel('48k'),
    DACModel(),
    CdpamModel('acoustic'), 
    CdpamModel('content')
]

res = {}
for model in models:
    try:
        model.load_model()
        metric = TimbreMetric(model, use_fadtk_model=True)
        res[model.name] = metric()
    except:
        print(f"Failed to load {model.name}")
        continue
print(res)
