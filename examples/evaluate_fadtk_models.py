import os
import torch
from fadtk.model_loader import *
from timbremetrics import (
    TimbreMetric,
    write_results_to_yaml,
)
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")


class DACModel_ours(ModelLoader):
    """our implementation for DAC following
        https://github.com/descriptinc/descript-audio-codec
    ---
    see fadtk's implementation:
        https://github.com/microsoft/fadtk/blob/main/fadtk/model_loader.py
    """

    def __init__(self):
        super().__init__("dac-44kHz", 1024, 44100)

    def load_model(self):
        from dac.utils import load_model

        self.model = load_model(tag="latest", model_type="44khz")
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio) -> torch.Tensor:

        audio = audio.to(self.device)
        x = self.model.preprocess(audio.audio_data, audio.sample_rate)
        z, _, _, _, _ = self.model.encode(x)
        z = z.squeeze(0)  # remove batch dimension
        z = z.T  # (n_features, n_frames) -> (n_frames, n_features)

        return z

    def load_wav(self, wav_file):
        from audiotools import AudioSignal

        return AudioSignal(wav_file)


models = [
    CLAPModel("2023"),
    CLAPLaionModel("audio"),
    CLAPLaionModel("music"),
    # VGGishModel(),
    # MERTModel(layer=6),
    EncodecEmbModel("24k"),
    EncodecEmbModel("48k"),
    # DACModel(),
    DACModel_ours(),
    CdpamModel("acoustic"),
    CdpamModel("content"),
    # W2V2Model("base", layer=6),
    # W2V2Model("large", layer=12),
    # HuBERTModel("base", layer=6),
    # HuBERTModel("large", layer=12),
    # WavLMModel("base", layer=6),
    # WavLMModel("base-plus", layer=6),
    # WavLMModel("large", layer=12),
    # WhisperModel('tiny'),
    # WhisperModel('small'),
    # WhisperModel("base"),
    # WhisperModel('medium'),
    # WhisperModel("large"),
]

err = {}
for model in models:
    try:
        model.load_model()

        if model.name in [
            "clap-2023",  # context window: 7s
            "clap-laion-audio",  # context window: 10s
            "clap-laion-music",  # context window: 10s
            "cdpam-acoustic",  # context window: 5s
            "cdpam-content",  # context window: 5s
        ]:

            metric = TimbreMetric(
                use_fadtk_model=True,
                fadtk_audio_loader=model.load_wav,
                fadtk_time_average=False,  # only take the first frame, no need to average over time
                sample_rate=model.sr,
                pad_to_the_longer_length=False,
            )
            res = metric(model)
            write_results_to_yaml(out_file, f"{model.name}", res)

        else:

            metric = TimbreMetric(
                use_fadtk_model=True,
                fadtk_audio_loader=model.load_wav,
                fadtk_time_average=True,  # average over time!
                sample_rate=model.sr,
                pad_to_the_longer_length=False,
            )
            res = metric(model)
            write_results_to_yaml(out_file, f"time-avg_{model.name}", res)

            metric = TimbreMetric(
                use_fadtk_model=True,
                fadtk_audio_loader=model.load_wav,
                fadtk_time_average=False,
                sample_rate=model.sr,
                pad_to_the_longer_length=True,
            )
            res = metric(model)
            write_results_to_yaml(out_file, model.name, res)

    except Exception as e:
        err[model.name] = str(e)
# print(err)
