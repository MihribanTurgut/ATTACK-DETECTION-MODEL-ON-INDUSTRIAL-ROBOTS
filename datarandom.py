import pandas as pd
import random

veri_seti = pd.read_csv('temizlenmis_dataset.csv')

sadece_sifirlar = veri_seti[veri_seti['binary attack'] == 0]

sifir_sayisi = len(sadece_sifirlar)
veri_sayisi = min(42000, sifir_sayisi)

rastgele_veri = sadece_sifirlar.sample(n=veri_sayisi, random_state=42)

print(rastgele_veri.head())
rastgele_veri.to_csv('randomdata.csv', index=False)

veri_seti2 = pd.read_csv('temizlenmis_dataset.csv')
sadece_birler = veri_seti2[veri_seti2['binary attack'] == 1]


bir_sayisi = len(sadece_birler)
veri_sayisi2 = min(42000, bir_sayisi)


rastgele_veri2 = sadece_birler.sample(n=veri_sayisi2, random_state=42)
print(rastgele_veri2.head())
rastgele_veri2.to_csv('randomdata1.csv', index=False)

df1=pd.read_csv('randomdata1.csv')
df2=pd.read_csv('randomdata.csv')

b=pd.concat([df1,df2], ignore_index=True)
print(b.head())
b.to_csv('allin60s.csv', index=False)

