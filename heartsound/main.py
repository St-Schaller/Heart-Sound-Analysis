import librosa.display
import pandas as pd
from matplotlib import pyplot as plt

from heartsound.data.preprocessing import split_audio, create_label_csv, create_csv, merge_labels, \
    split_test_training_labels
from sklearn.metrics import accuracy_score, classification_report

#split_audio('/home/local/Dokumente/HeartApp/PASCAL/AB_training', '/home/local/Dokumente/HeartApp/PASCAL/AB_training_splitted', 4)

#create_label_csv('/home/local/Dokumente/HeartApp/physionet_challenge/splitted_wav', '/home/local/Dokumente/HeartApp/physionet_challenge/')
#create_label_csv('/home/local/Dokumente/HeartApp/PASCAL/AB_training_splitted', '/home/local/Dokumente/HeartApp/PASCAL/')
#create_csv('/home/local/Dokumente/HeartApp/All_Data', '/home/local/Dokumente/HeartApp/')
#merge_labels()

#split_test_training_labels()


'''data, samrate = librosa.load('/home/local/Dokumente/HeartApp/physionet_challenge/splitted_wav/a0001_splitted_0', sr=4000)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=samrate)
plt.show()

data, samrate = librosa.load('/home/local/Dokumente/HeartApp/physionet_challenge/wav_combined/a0001.wav', sr=4000)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=samrate)
plt.show()
'''

predictions = pd.read_csv('/home/local/Dokumente/HeartApp/prediction_potes_test.csv')
print(predictions)
predictions.columns = ['filename', 'predictions']
#predictions = predictions.rename(columns={0: "filename", 1: 'labels'})
#data_np.insert(0, 'file_name', names)
print(predictions)
labels = pd.read_csv('/home/local/Dokumente/HeartApp/test_labels_all_data_combined.csv', names=['filename', 'labels'])
print(labels)

df_merged = pd.merge(predictions, labels, how='inner', on='filename')
print(df_merged)

print(accuracy_score(df_merged['labels'], df_merged['predictions']))
print(classification_report(df_merged['labels'], df_merged['predictions']))

