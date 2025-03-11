from fadtk.model_loader import *
from timbremetrics.metrics import TimbreMetric
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

models = [
    CLAPModel("2023"),
    CLAPLaionModel("audio"),
    CLAPLaionModel("music"),
    EncodecEmbModel("24k"),
    EncodecEmbModel("48k"),
    DACModel(),
    CdpamModel("acoustic"),
    CdpamModel("content"),
    W2V2Model("base", layer=10),
    HuBERTModel("base", layer=10),
    WavLMModel("base", layer=10),
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
print(res)
print(err)
