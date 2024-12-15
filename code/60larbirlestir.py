import pandas as pd
import random

veri_60_sifir = pd.read_csv('60sifir.csv')
veri_60_bir = pd.read_csv('60bir.csv')

birlesik_veri = pd.concat([veri_60_sifir, veri_60_bir])
birlesik_veri.to_csv('birlesik_veri.csv', index=False)