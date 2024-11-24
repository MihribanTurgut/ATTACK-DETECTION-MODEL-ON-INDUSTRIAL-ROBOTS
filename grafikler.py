import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#atak grafiği sayısı
data_multi = pd.read_csv('ROSIDS23-multi.csv')

attack_counts = data_multi['Label'].value_counts()


for attack_type, count in attack_counts.items():
    print(f"{attack_type}: {count}")


plt.figure(figsize=(12, 6))
sns.barplot(x=attack_counts.index, y=attack_counts.values, palette='viridis')

plt.title('Number of Attack Types', color='black')
plt.xlabel('Attack Types', color='black')
plt.ylabel('Number of Attack', color='black')
plt.xticks(rotation=45, color='black')
plt.yticks(color='black')
plt.show()


#pie chart for 60s
data_guncel = pd.read_csv('guncel.csv')
data_birlesik = pd.read_csv('birlesik_veri.csv')

binary_counts_guncel = data_guncel['Attack Binary'].value_counts().sort_index()


total_zeros = (data_birlesik['Attack Binary'] == 0).sum()
total_ones = (data_birlesik['Attack Binary'] == 1).sum()

binary_counts_birlesik = pd.Series([total_zeros, total_ones], index=[0, 1])


def absolute_value(val):
    return int(val/100.*sum(binary_counts_guncel))


def absolute_value_birlesik(val):
    return int(val/100.*sum(binary_counts_birlesik))


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
binary_counts_guncel.plot(
    kind='pie', 
    autopct=lambda p: absolute_value(p), 
    colors=['pink', 'purple'], 
    startangle=90, 
    counterclock=False,
    textprops={'fontsize': 17}  
)
plt.title(
    'Distribution of Raw Data', 
    fontsize=16, 
    color='purple',
    bbox=dict(facecolor='lavender', edgecolor='purple', boxstyle='round,pad=0.5')
)
plt.ylabel('')

# Second pie chart
plt.subplot(1, 2, 2)
binary_counts_birlesik.plot(
    kind='pie', 
    autopct=lambda p: absolute_value_birlesik(p), 
    colors=['pink', 'purple'], 
    startangle=90, 
    counterclock=False,
    textprops={'fontsize': 17} 
)
plt.title(
    'Distribution of 60,000 - 60,000 Split Data', 
    fontsize=16, 
    color='purple',
    bbox=dict(facecolor='lavender', edgecolor='purple', boxstyle='round,pad=0.5')
)
plt.ylabel('')

plt.tight_layout()
plt.show()
