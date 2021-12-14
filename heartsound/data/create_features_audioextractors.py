import glob
from pathlib import Path
import numpy as np
import pandas as pd
import openl3
import soundfile as sf



def extract_data_openl3(inputfolder,labeldata, outputpath):
    data = []
    names = []

    colnames = ['filename', 'labels']
    labels = pd.read_csv(labeldata, names=colnames,
                         header=None)
    print(labels)

    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="env", embedding_size=6144)

    for audio_path in sorted(glob.iglob(inputfolder + '/*.wav')):
        emb_list = []
        audio, sr = sf.read(audio_path)
        emb, ts = openl3.get_audio_embedding(audio, sr, model=model, hop_size=0.5)
        file = Path(audio_path).stem
        names.append(file)
        print(file)

        emb_list = calculate_mean_vector(emb)
        data.append(emb_list)

        # if you want to store raw embeddings with timestamps
        # output_path = os.path.join(dir, file + '_openl3.npz')
        # np.savez(output_path, embedding=emb, timestamps=ts)

    data_np = pd.DataFrame(data)
    data_np.insert(0, 'filename', names)
    print(data_np)
    df_merged = pd.merge(data_np, labels, how='inner', on='filename')
    print(df_merged)
    df_merged.to_csv(outputpath + 'filename', index=False)

def calculate_mean_vector(emb_list):
    emb_list = np.array(emb_list)
    mean_vector = emb_list.mean(0)
    return mean_vector