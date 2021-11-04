import glob
from pathlib import Path
import pandas as pd
import openl3
import soundfile as sf



def extract_data_openl3():
    data = []
    names = []

    colnames = ['file_name', 'labels']
    # labels = pd.read_table('/home/local/Dokumente/CI_14/Snore_dist/lab/ComParE2017_Snore_test.tsv')
    labels = pd.read_csv('/home/local/Dokumente/HeartApp/physionet_challenge/labels_combined.csv', names=colnames,
                         header=None)
    print(labels)

    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="env", embedding_size=6144)

    # for audio_path in sorted(glob.iglob('/home/local/Dokumente/HeartApp/physionet_challenge/wav_combined/*.wav')):
    for audio_path in sorted(glob.iglob('/home/local/Dokumente/HeartApp/physionet_challenge/validation/*.wav')):
        emb_list = []
        audio, sr = sf.read(audio_path)
        emb, ts = openl3.get_audio_embedding(audio, sr, model=model, hop_size=0.5)
        file = Path(audio_path).stem
        # file = os.path.basename(audio_path)
        names.append(file)
        print(file)

        # add all embeddings together instead of mean vector
        # for embedding in emb:
        # emb_list.append(embedding)
        # emb_list = np.vstack(emb_list).flatten()

        emb_list = calculate_mean_vector(emb)
        data.append(emb_list)

        # if you want to store raw embeddings with timestamps
        # output_path = os.path.join(dir, file + '_openl3.npz')
        # np.savez(output_path, embedding=emb, timestamps=ts)

    data_np = pd.DataFrame(data)
    data_np.insert(0, 'file_name', names)
    print(data_np)
    df_merged = pd.merge(data_np, labels, how='inner', on='file_name')
    print(df_merged)
    path = '/home/local/Dokumente/HeartApp/physionet_challenge/'
    df_merged.to_csv(path + 'Physionet_validation_05hop_meanvector_6144_openl3.csv', index=False)