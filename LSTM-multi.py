import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

test = pd.read_csv('test1.csv')
train = pd.read_csv('train1.csv')
test.drop(columns=['Label'], inplace=True)
train.drop(columns=['Label'], inplace=True)

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
    '8.6.0.1': 21,
    '8.0.6.4': 22,
    '224.0.0.7': 23,
    '239.255.255.250': 24,
    '255.255.255.255': 25,
    '224.0.0.251': 26
}

train['Src IP'] = train['Src IP'].replace(ip_mapping)
train['Dst IP'] = train['Dst IP'].replace(ip_mapping)

test['Src IP'] = test['Src IP'].replace(ip_mapping)
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
    'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Timestamp', 'Attack'
]

train['Flow Byts/s'] = train['Flow Byts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
train['Flow Pkts/s'] = train['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
test['Flow Byts/s'] = test['Flow Byts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)
test['Flow Pkts/s'] = test['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan).fillna(0)

train = train[features]
test = test[features]

X_train = train.drop(columns=['Timestamp', 'Attack'])
y_train = train['Attack']

X_test = test.drop(columns=['Timestamp', 'Attack'])
y_test = test['Attack']

# One-hot encoding multi bir veri olduğu gerekçesiyle
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))

# StandardScaler and MinMaxScaler
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
X_train_scaled_std = scaler1.fit_transform(X_train)
X_test_scaled_std = scaler1.transform(X_test)
X_train_scaled_mm = scaler2.fit_transform(X_train)
X_test_scaled_mm = scaler2.transform(X_test)
#minmax kullanılmadı denendi

# Random Forest 
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled_std, y_train)

feature_importances_rf = rf_model.feature_importances_

top_indices_rf = np.argsort(feature_importances_rf)[::-1][:45]
top_features_rf = X_train.columns[top_indices_rf]

# Mutual 
mutual_info_scores = mutual_info_classif(X_train_scaled_std, y_train)
top_indices_mi = np.argsort(mutual_info_scores)[::-1][:45]
top_features_mi = X_train.columns[top_indices_mi]

#common
common_features = set(top_features_rf).intersection(top_features_mi)

X_train_common = X_train[list(common_features)]
X_test_common = X_test[list(common_features)]


scaler_common = StandardScaler()
X_train_scaled_common = scaler_common.fit_transform(X_train_common)
X_test_scaled_common = scaler_common.transform(X_test_common)


# LSTM 
model_common = Sequential()
model_common.add(LSTM(units=64, return_sequences=True, input_shape=(X_train_scaled_common.shape[1], 1)))
model_common.add(Dropout(0.2))
model_common.add(LSTM(units=128, return_sequences=True))
model_common.add(Dropout(0.2))
model_common.add(LSTM(units=128))
model_common.add(Dense(units=128, activation='relu'))
model_common.add(Dropout(0.2))
model_common.add(Dense(y_train_encoded.shape[1], activation='softmax'))

model_common.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_common.summary()
model_common.save('LSTM-multi.h5')
history_common = model_common.fit(X_train_scaled_common, y_train_encoded, epochs=150, batch_size=128, validation_split=0.2)

score_common = model_common.evaluate(X_test_scaled_common, y_test_encoded, verbose=0)
print('LSTM Test loss:', score_common[0])
print('LSTM Test accuracy :', score_common[1])


y_pred_common = model_common.predict(X_test_scaled_common)
y_pred_common_classes = np.argmax(y_pred_common, axis=1)
y_test_classes = np.argmax(y_test_encoded, axis=1)


precision = precision_score(y_test_classes, y_pred_common_classes, average='weighted')
recall = recall_score(y_test_classes, y_pred_common_classes, average='weighted')
f1 = f1_score(y_test_classes, y_pred_common_classes, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

plt.plot(history_common.history['accuracy'], label='Train Accuracy ')
plt.plot(history_common.history['val_accuracy'], label='Validation Accuracy ')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history_common.history['loss'], label='Train Loss ')
plt.plot(history_common.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
