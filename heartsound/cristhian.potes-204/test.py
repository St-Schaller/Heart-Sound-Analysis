import glob
from pathlib import Path

import matlab.engine
import pandas as pd

eng = matlab.engine.start_matlab()
labels_potes = []
for recordName in sorted(glob.glob("/home/local/Dokumente/HeartApp/All_Data/All_Data_test/*.wav")):
    print(recordName)
    result = eng.challenge(recordName)
    print(result)
    labels_potes.append([Path(recordName).stem, result])

results_np = pd.DataFrame(labels_potes)
results_np = results_np.rename(index={0: 'filename', 1: 'labels'})
print(results_np)
path = '/home/local/Dokumente/HeartApp/'
results_np.to_csv(path + 'prediction_potes_test.csv', index=False)