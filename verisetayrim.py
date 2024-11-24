import pandas as pd
from sklearn.model_selection import train_test_split

#bölünme 30 a 70
veri_seti = pd.read_csv('ROSIDS23-multi.csv')
veri_seti_egitim, veri_seti_test = train_test_split(veri_seti, test_size=0.3, random_state=42)
veri_seti_egitim.to_csv('train1.csv', index=False)
veri_seti_test.to_csv('test1.csv', index=False)