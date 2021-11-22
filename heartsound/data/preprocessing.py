import glob

import pandas as pd
from pydub import AudioSegment
from pathlib import Path


def split_audio(input_folder, output_folder, sec_per_split):
    for filepath in sorted(glob.iglob( input_folder +'/*.wav')):
        audio = AudioSegment.from_wav(filepath)
        total_sec = int(audio.duration_seconds)
        filename = Path(filepath).stem
        for i in range(0, total_sec, sec_per_split):
            split_fn = filename + '_splitted_' + str(i)
            t1 = i * 1000
            t2 = (i + sec_per_split) * 1000
            split_audio = audio[t1:t2]
            split_audio.export(output_folder + '/' + split_fn, format="wav")
            print(str(split_fn) + ' Done')
            if i == total_sec - sec_per_split:
                print('All splited successfully')

def create_label_csv(input_folder, output_folder):
    table = []
    for filepath in sorted(glob.iglob(input_folder + '/*.wav')):

        filename = Path(filepath).stem
        if 'normal' in filename:
            label = -1
        else:
            label = 1
        table.append([filename, label])
    df = pd.DataFrame(table, columns=['filename', 'label'])
    print(df)
    df.to_csv(output_folder + 'label_pascal.csv', index=False)




def create_csv(input_folder, output_folder):
    table = []
    for filepath in sorted(glob.iglob(input_folder + '/*.wav')):

        filename = Path(filepath).stem
        table.append([filename])
    df = pd.DataFrame(table, columns=['filename'])
    print(df)
    df.to_csv(output_folder + 'labels_all_data_combined.csv', index=False)


def merge_labels():
    df = pd.read_csv('/home/local/Dokumente/HeartApp/labels_all_data_splitted_combined.csv');
    labels1 = pd.read_csv('/home/local/Dokumente/HeartApp/PASCAL/label_pascal_splitted.csv');
    print(labels1)
    labels2 = pd.read_csv('/home/local/Dokumente/HeartApp/physionet_challenge/labels_physionet_splitted.csv', names=['filename','label']);


    print(labels2)
    all_labels = pd.concat([labels1, labels2], ignore_index=True)
    print(all_labels)
    df = df.merge(all_labels, how='left', on=['filename'])
    print(df)
    df.to_csv('/home/local/Dokumente/HeartApp/labels_all_data_splitted_combined.csv', index=False)


def split_test_training_labels():
    df = pd.read_csv('/home/local/Dokumente/HeartApp/labels_all_data_combined.csv');
    df_train_1 = df.iloc[:970,:]
    df_train_2 = df.iloc[1025:3231,:]
    df_train_3 = df.iloc[3345:, :]
    df_train = pd.concat([df_train_1, df_train_2, df_train_3], ignore_index=True)
    df_test_1 = df.iloc[970:1025,:]
    df_test_2 = df.iloc[3231:3345, :]
    df_test = pd.concat([df_test_1,df_test_2], ignore_index=True)
    print(df_train)
    print(df_test)
    df_train.to_csv('/home/local/Dokumente/HeartApp/training_labels_all_data_combined.csv', index=False)
    df_test.to_csv('/home/local/Dokumente/HeartApp/test_labels_all_data_combined.csv', index=False)










