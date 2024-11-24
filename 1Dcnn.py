import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.utils import plot_model

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


ip_mapping = {
    '192.168.3.1': 1, '192.168.3.2': 2, '192.168.3.3': 3, '192.168.3.4': 4, '192.168.3.5': 5,
    '192.168.3.6': 6, '192.168.3.7': 7, '192.168.3.8': 8, '192.168.3.9': 9, '192.168.3.10': 10,
    '192.168.3.11': 11, '192.168.3.12': 12, '192.168.3.13': 13, '192.168.3.14': 14, '192.168.3.15': 15,
    '192.168.3.16': 16, '192.168.3.17': 17, '192.168.3.18': 18, '192.168.3.19': 19, '192.168.3.20': 20,
    '8.6.0.1': 21, '8.0.6.4': 22, '224.0.0.7': 23, '239.255.255.250': 24, '255.255.255.255': 25,
    '224.0.0.251': 26
}

train['Src IP'] = train['Src IP'].replace(ip_mapping)
train['Dst IP'] = train['Dst IP'].replace(ip_mapping)
test['Src IP'] = test['Src IP'].replace(ip_mapping)
test['Dst IP'] = test['Dst IP'].replace(ip_mapping)


features = [
    'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
    'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
    'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Bwd PSH Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Max', 'Pkt Len Mean',
    'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
    'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Bwd Byts/b Avg',
    'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Fwd Act Data Pkts', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Timestamp', 'Attack Binary'
]


train['Flow Byts/s'] = train['Flow Byts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
train['Flow Pkts/s'] = train['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
test['Flow Byts/s'] = test['Flow Byts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
test['Flow Pkts/s'] = test['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)


train = train[features]
test = test[features]

X_train = train.drop(columns=['Timestamp', 'Attack Binary'])
y_train = train['Attack Binary']
X_test = test.drop(columns=['Timestamp', 'Attack Binary'])
y_test = test['Attack Binary']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
feature_importances_rf = rf_model.feature_importances_
top_indices_rf = np.argsort(feature_importances_rf)[::-1][:25]
top_features_rf = X_train.columns[top_indices_rf]


mutual_info_scores = mutual_info_classif(X_train_scaled, y_train)
top_indices_mi = np.argsort(mutual_info_scores)[::-1][:25]
top_features_mi = X_train.columns[top_indices_mi]

common_features = set(top_features_rf).intersection(top_features_mi)

X_train_common = X_train[list(common_features)]
X_test_common = X_test[list(common_features)]

scaler_common = StandardScaler()
X_train_scaled_common = scaler_common.fit_transform(X_train_common)
X_test_scaled_common = scaler_common.transform(X_test_common)


X_train_scaled_common = X_train_scaled_common.reshape((X_train_scaled_common.shape[0], X_train_scaled_common.shape[1], 1))
X_test_scaled_common = X_test_scaled_common.reshape((X_test_scaled_common.shape[0], X_test_scaled_common.shape[1], 1))


model_common = Sequential()
model_common.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled_common.shape[1], 1)))
model_common.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model_common.add(MaxPooling1D(pool_size=2))
model_common.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model_common.add(Dropout(0.5))
model_common.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model_common.add(MaxPooling1D(pool_size=2))
model_common.add(Flatten())
model_common.add(Dense(128, activation='relu'))
model_common.add(Dropout(0.5))
model_common.add(Dense(64, activation='relu'))
model_common.add(Dense(1, activation='sigmoid'))

model_common.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_common.summary()
model_common.save('CNN.h5')

history_common = model_common.fit(X_train_scaled_common, y_train, epochs=100, batch_size=64, validation_split=0.2)

score_common = model_common.evaluate(X_test_scaled_common, y_test, verbose=0)
print('1D CNN Test loss with common features:', score_common[0])
print('1D CNN Test accuracy with common features:', score_common[1])

y_pred_encoded = model_common.predict(X_test_scaled_common)
y_pred = (y_pred_encoded > 0.5).astype(int)
y_test_decoded = y_test

precision = precision_score(y_test_decoded, y_pred, average='weighted')
recall = recall_score(y_test_decoded, y_pred, average='weighted')
f1 = f1_score(y_test_decoded, y_pred, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_common.history['accuracy'])
plt.plot(history_common.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_common.history['loss'])
plt.plot(history_common.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
