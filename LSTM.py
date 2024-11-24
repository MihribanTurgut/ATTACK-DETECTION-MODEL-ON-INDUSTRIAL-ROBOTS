import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

ip_mapping = {
    '192.168.3.1': 1,
    '192.168.3.2': 2,
    '192.168.3.3': 3,
    '192.168.3.4': 4,
    '192.168.3.5': 5,
    '192.168.3.6': 6,
    '192.168.3.7': 7,
    '192.168.3.8': 8,
    '192.168.3.9': 9,
    '192.168.3.10': 10,
    '192.168.3.11': 11,
    '192.168.3.12': 12,
    '192.168.3.13': 13,
    '192.168.3.14': 14,
    '192.168.3.15': 15,
    '192.168.3.16': 16,
    '192.168.3.17': 17,
    '192.168.3.18': 18,
    '192.168.3.19': 19,
    '192.168.3.20': 20,
    '8.6.0.1':21,
    '8.0.6.4':22,
    '224.0.0.7':23,
    '239.255.255.250':24,
    '255.255.255.255':25,
    '224.0.0.251':26
}

train['Src IP'] = train['Src IP'].replace(ip_mapping)
train['Dst IP'] = train['Dst IP'].replace(ip_mapping)

test['Src IP']= test['Src IP'].replace(ip_mapping)
test['Dst IP'] = test['Dst IP'].replace(ip_mapping)


features = [
    'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Flow Duration',
    'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Bwd IAT Min', 'Bwd PSH Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'Bwd Byts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
    'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Fwd Act Data Pkts', 'Active Mean',
    'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Attack Binary', 'Timestamp'
]

train['Flow Byts/s'] = train['Flow Byts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
train['Flow Pkts/s'] = train['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
test['Flow Byts/s'] = test['Flow Byts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
test['Flow Pkts/s'] = test['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)

train = train[features]
test = test[features]

X_train = train.drop(columns=['Attack Binary','Timestamp'])
y_train = train['Attack Binary']

X_test = test.drop(columns=['Attack Binary','Timestamp'])
y_test = test['Attack Binary']

#mutual classif
mutual_importances = mutual_info_classif(X_train, y_train)

feature_importances_mutual = pd.DataFrame(mutual_importances, index=X_train.columns, columns=['Importance'])
feature_importances_mutual = feature_importances_mutual.sort_values(by='Importance', ascending=False)

top_features_mutual = feature_importances_mutual.head(45)
print(top_features_mutual)  

#random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importances = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=['Importance'])
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
top_features = feature_importances.head(45)  
print(top_features)

#common feature
common_features = top_features_mutual.index.intersection(top_features.index)
common_features_df = pd.DataFrame(index=common_features)
print(common_features_df)

top_features_columns = common_features_df.index.tolist()
X_train_filtered = X_train[top_features_columns]
X_test_filtered = X_test[top_features_columns]

scaler1 = StandardScaler()
X_train_scaled_filtered = scaler1.fit_transform(X_train_filtered)
X_test_scaled_filtered = scaler1.transform(X_test_filtered)


#bilstm model
model = Sequential()
model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train_scaled_filtered.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=128)))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#model.save('BiLSTM.h5')
#early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(X_train_scaled_filtered, y_train, epochs=150, batch_size=128, validation_split=0.2)
 
score = model.evaluate(X_test_scaled_filtered, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(X_test_scaled_filtered)
y_pred = (y_pred > 0.5) 


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#scaled
sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 7))

#Unscaled data
plt.subplot(1, 2, 1)
plt.hist(X_train_filtered.values.flatten(), bins=50, color='yellow', alpha=0.7, edgecolor='black')
plt.title('Unscaled Data Distribution', fontsize=18, color='black')
plt.xlabel('Value', fontsize=14, color='black')
plt.ylabel('Frequency', fontsize=14, color='black')
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(['Unscaled'], fontsize=12)

#Scaled data
plt.subplot(1, 2, 2)
plt.hist(X_train_scaled_filtered.flatten(), bins=50, color='orange', alpha=0.7, edgecolor='black')
plt.title('Scaled Data Distribution', fontsize=18, color='black')
plt.xlabel('Value', fontsize=14, color='black')
plt.ylabel('Frequency', fontsize=14, color='black')
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(['Scaled'], fontsize=12)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sorted_features = feature_importances.loc[common_features].sort_values(by='Importance', ascending=True)  

sns.set_theme(style="whitegrid")
sns.barplot(x=sorted_features['Importance'], y=sorted_features.index, palette='viridis')

plt.xlabel('Importance', fontsize=14, color='black')
plt.ylabel('Common Features', fontsize=14, color='black')
plt.title('Feature Importances of Common Features (Low to High)', fontsize=16, color='black')
plt.xticks(color='black', fontsize=12)
plt.yticks(color='black', fontsize=12)

plt.show()
