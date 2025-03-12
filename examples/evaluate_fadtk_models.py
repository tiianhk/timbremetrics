import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from fadtk.model_loader import *
from timbremetrics.metrics import TimbreMetric

models = [
    CLAPModel("2023"),
    CLAPLaionModel("audio"),
    CLAPLaionModel("music"),
    VGGishModel(),
    MERTModel(layer=12),
    EncodecEmbModel("24k"),
    EncodecEmbModel("48k"),
    DACModel(),
    CdpamModel("acoustic"),
    CdpamModel("content"),
    W2V2Model("base", layer=12),
    W2V2Model("large", layer=24),
    HuBERTModel("base", layer=12),
    HuBERTModel("large", layer=24),
    WavLMModel("base", layer=12),
    WavLMModel("base-plus", layer=12),
    WavLMModel("large", layer=24),
    WhisperModel("tiny"),
    WhisperModel("small"),
    WhisperModel("base"),
    WhisperModel("medium"),
    WhisperModel("large"),
]

res = {}
err = {}
for model in models:
    try:
        model.load_model()
        metric = TimbreMetric(
            model,
            use_fadtk_model=True,
            fadtk_keep_time_dimension=True,
            pad_to_max_duration=True,
        )
        res[model.name] = metric()
    except Exception as e:
        err[model.name] = str(e)
    torch.cuda.empty_cache()
print(res)
print(err)
