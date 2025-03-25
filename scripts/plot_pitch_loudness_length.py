import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pyloudnorm as pyln

from timbremetrics import AudioLoader
from timbremetrics.paths import BASE_DIR

audio_loader = AudioLoader()
datasets = audio_loader._load_audio_datasets()
loudness = {}
audio_length = {}
for d in datasets:
    loudness[d] = []
    audio_length[d] = []
    for x in datasets[d]:
        audio, sr = x["audio"], x["sample_rate"]
        meter = pyln.Meter(sr, block_size=0.08)
        loudness[d].append(meter.integrated_loudness(audio.numpy().T))
        audio_length[d].append(audio.shape[-1] / sr)

pitch = {
    "Barthet2010": [165] * 15,
    "Grey1977": [311] * 16,
    "Grey1978": [311] * 16,
    "Iverson1993_Onset": [262] * 16,
    "Iverson1993_Remainder": [262] * 16,
    "Iverson1993_Whole": [262] * 16,
    "Lakatos2000_Comb": [311] * 15,  # 5 are weakly- or non-pitched
    "Lakatos2000_Harm": [311] * 17,
    "Lakatos2000_Perc": [311] * 7,  # 11 are weakly- or non-pitched
    "McAdams1995": [311] * 18,
    "Patil2012_A3": [220] * 11,
    "Patil2012_DX4": [294] * 11,
    "Patil2012_GD4": [415] * 11,
    "Saitis2020_e2set1_general": [311] * 14,
    "Siedenburg2016_e2set1": [311] * 14,
    "Siedenburg2016_e2set2": [311] * 14,
    "Siedenburg2016_e2set3": [311] * 14,
    "Siedenburg2016_e3": [311] * 14,
    "Vahidi2020": [440] * 15,
    "Zacharakis2014_english": [55] * 1 + [110] * 6 + [220] * 14 + [440] * 3,
    "Zacharakis2014_greek": [55] * 1 + [110] * 6 + [220] * 14 + [440] * 3,
}  # from paper


def dict_to_df(data_dict, feature_name):
    data = []
    for dataset, values in data_dict.items():
        for value in values:
            data.append({"Dataset": dataset, feature_name: value})
    return pd.DataFrame(data)


df_pitch = dict_to_df(pitch, "Pitch (Hz)")
df_loudness = dict_to_df(loudness, "Loudness (dB)")
df_audio_length = dict_to_df(audio_length, "Audio Length (s)")

# plot
sns.set(style="darkgrid")
fig, axes = plt.subplots(
    3, 1, figsize=(8, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)

# pitch
sns.stripplot(x="Dataset", y="Pitch (Hz)", data=df_pitch, ax=axes[0], color="#0077BB")
pitch_values = sorted(set(df_pitch["Pitch (Hz)"]))
axes[0].set_yticks(pitch_values)
axes[0].set_yticklabels([f"{int(v)}" for v in pitch_values])
axes[0].set_ylabel("Pitch (Hz)", fontsize=16)

# loudness
sns.stripplot(
    x="Dataset", y="Loudness (dB)", data=df_loudness, ax=axes[1], color="#009988"
)
axes[1].set_ylabel("Loudness (dB)", fontsize=16)

# audio length
sns.stripplot(
    x="Dataset", y="Audio Length (s)", data=df_audio_length, ax=axes[2], color="#EE7733"
)
axes[2].set_ylabel("Length (seconds)", fontsize=16)

axes[-1].set_xlabel("")
plt.xticks(rotation=90, fontsize=16)
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "../assets/pitch-loudness-length.png"),
    bbox_inches="tight",
    dpi=300,
)
