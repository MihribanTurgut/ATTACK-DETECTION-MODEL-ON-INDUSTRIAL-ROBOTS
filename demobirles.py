import pandas as pd
import random


#60 ayrılan 0 ve 1 leri 120k olacak şekilde tek csv
#daha sonra csv test ve train bölünecek
veri_seti = pd.read_csv('guncel.csv')

sadece_sifirlar = veri_seti[veri_seti['Attack Binary'] == 0]

sifir_sayisi = len(sadece_sifirlar)
veri_sayisi = min(5000, sifir_sayisi)

rastgele_veri = sadece_sifirlar.sample(n=veri_sayisi, random_state=42)

print(rastgele_veri.head())
rastgele_veri.to_csv('5sifir.csv', index=False)

veri_seti2 = pd.read_csv('guncel.csv')
sadece_birler = veri_seti2[veri_seti2['Attack Binary'] == 1]


bir_sayisi = len(sadece_birler)
veri_sayisi2 = min(5000, bir_sayisi)


rastgele_veri2 = sadece_birler.sample(n=veri_sayisi2, random_state=42)
print(rastgele_veri2.head())
rastgele_veri2.to_csv('5bir.csv', index=False)


veri_60_sifir = pd.read_csv('5sifir.csv')
veri_60_bir = pd.read_csv('5bir.csv')


birlesik_veri = pd.concat([veri_60_sifir, veri_60_bir])
birlesik_veri.to_csv('traindemo.csv', index=False)
