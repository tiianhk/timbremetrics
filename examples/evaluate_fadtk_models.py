import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fadtk.model_loader import *
from timbremetrics.metrics import TimbreMetric
from timbremetrics.utils import print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")

models = [
    CLAPModel("2023"),
    CLAPLaionModel("audio"),
    CLAPLaionModel("music"),
    VGGishModel(),
    MERTModel(layer=6),
    EncodecEmbModel("24k"),
    EncodecEmbModel("48k"),
    DACModel(),
    CdpamModel("acoustic"),
    CdpamModel("content"),
    # W2V2Model("base", layer=6),
    W2V2Model("large", layer=12),
    # HuBERTModel("base", layer=6),
    HuBERTModel("large", layer=12),
    # WavLMModel("base", layer=6),
    # WavLMModel("base-plus", layer=6),
    WavLMModel("large", layer=12),
    # WhisperModel('tiny'),
    # WhisperModel('small'),
    # WhisperModel("base"),
    # WhisperModel('medium'),
    WhisperModel("large"),
]

err = {}
for model in models:
    try:
        model.load_model()

        metric = TimbreMetric(
            model,
            use_fadtk_model=True,
            fadtk_keep_time_dimension=False,
            pad_to_max_duration=False,
        )
        res = metric()
        write_results_to_yaml(out_file, model.name + ", keep_time_dimension=False", res)

        metric = TimbreMetric(
            model,
            use_fadtk_model=True,
            fadtk_keep_time_dimension=True,
            pad_to_max_duration=True,
        )
        res = metric()
        write_results_to_yaml(out_file, model.name + ", keep_time_dimension=True", res)

    except Exception as e:
        err[model.name] = str(e)
print(err)
