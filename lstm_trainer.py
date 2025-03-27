import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras_tuner.tuners import BayesianOptimization

# Konstanten
NUM_PRODUCT_GROUPS = 17
MAX_SEQUENCE_LENGTH = 60

# CSV laden
df = pd.read_csv('input.csv', delimiter=',')

# Ordne Gruppe neuem Bereich [0, 17] basierend auf sortierten IDs zu
unique_ids = sorted(df['Gruppe'].unique())
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

# Mapping anwenden
df['Gruppe'] = df['Gruppe'].map(id_mapping)

scaler = MinMaxScaler()

# Wende Min-Max-Skalierung auf Wetter- und Nachfragedaten an
df[['cli_dwd_temperatur', 'cli_dwd_niederschlag', 'Menge']] = scaler.fit_transform(
    df[['cli_dwd_temperatur', 'cli_dwd_niederschlag', 'Menge']]
)

# Zyklische Kodierung für den Monat
df['Monat_sin'] = np.sin(2 * np.pi * df['Monat'] / 12)
df['Monat_cos'] = np.cos(2 * np.pi * df['Monat'] / 12)

# Entferne die 'Monat'-Reihe da sie nicht mehr gebraucht wird
df = df.drop(columns=['Monat'])

years = sorted(df['Year'].unique())
train_years = years[:6]
val_years = [years[6]]
test_years = [years[7]]

train_df = df[df['Year'].isin(train_years)]
test_df = df[df['Year'].isin(test_years)]
val_df = df[df['Year'].isin(val_years)]


# Funktion zum Erstellen von Sequenzen (Sliding Window)
def create_sequences(data, max_seq_length):
    X_timeseries, X_product, y = [], [], []
    grouped = data.groupby(['Gruppe', 'Year'])
    for (grp, yr), group in grouped:
        group = group.sort_values('Datum').reset_index(drop=True)
        if len(group) > max_seq_length:
            for i in range(len(group) - max_seq_length):
                seq = group.iloc[i:i + max_seq_length]
                target = group.iloc[i + max_seq_length]['Menge']
                features = seq[['Monat_sin', 'Monat_cos', 'cli_dwd_temperatur', 'cli_dwd_niederschlag']].values
                X_timeseries.append(features)
                # Produktgruppe als separate Input-Variable (als Integer)
                X_product.append(group.iloc[i]['Gruppe'])
                y.append(target)
    return np.array(X_timeseries), np.array(X_product), np.array(y).reshape(-1, 1)


# Sequenzen für alle Splits erstellen
X_train_ts, X_train_prod, y_train = create_sequences(train_df, MAX_SEQUENCE_LENGTH)
X_val_ts, X_val_prod, y_val = create_sequences(val_df, MAX_SEQUENCE_LENGTH)
X_test_ts, X_test_prod, y_test = create_sequences(test_df, MAX_SEQUENCE_LENGTH)


# Definition eigener Metriken
def smape(y_true, y_pred):
    y_true_orig = y_true
    y_pred_orig = y_pred

    epsilon = keras.backend.epsilon()
    numerator = tf.abs(y_true_orig - y_pred_orig)
    denominator = tf.abs(y_true_orig) + tf.abs(y_pred_orig) + epsilon
    return tf.reduce_mean(2.0 * numerator / denominator) * 100


def r2_metric(y_true, y_pred):
    y_true_orig = y_true
    y_pred_orig = y_pred

    ss_res = tf.reduce_sum(tf.square(y_true_orig - y_pred_orig))
    ss_tot = tf.reduce_sum(tf.square(y_true_orig - tf.reduce_mean(y_true_orig)))
    return 1 - ss_res / (ss_tot + keras.backend.epsilon())


# Modell-Building-Funktion mit Hyperparametern
def build_model(hp):
    # Hyperparameter
    seq_len = hp.Int("sequence_length", min_value=5, max_value=60, step=5)
    lstm_units = hp.Int("lstm_units", min_value=16, max_value=512, step=16)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    embedding_size = hp.Int("embedding_size", min_value=2, max_value=50, step=1)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")

    # Inputs: Die Zeitreihen-Daten haben festen Input (MAX_SEQUENCE_LENGTH, 4)
    timeseries_input = keras.Input(shape=(MAX_SEQUENCE_LENGTH, 4), name="timeseries_input")
    product_group_input = keras.Input(shape=(1,), name="product_group_input")

    # Lambda-Schicht: Slicing der letzten seq_len Zeitschritte
    x_sliced = layers.Lambda(lambda t: t[:, -seq_len:, :])(timeseries_input)
    # LSTM-Zweig
    x = layers.LSTM(lstm_units, dropout=dropout_rate)(x_sliced)

    # Embedding-Zweig für die Produktgruppe
    embedded = layers.Embedding(input_dim=NUM_PRODUCT_GROUPS, output_dim=embedding_size)(product_group_input)
    embedded = layers.Flatten()(embedded)

    # Zusammenführen der Zweige
    concatenated = layers.concatenate([x, embedded])
    dense = layers.Dense(64, activation="relu")(concatenated)
    output = layers.Dense(1, activation="linear", name="output")(dense)

    model = keras.Model(inputs=[timeseries_input, product_group_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse",
                  metrics=["mae", smape, r2_metric])
    return model


# Bayesian Optimization Tuner einrichten
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='lstm_bayes_dir',
    project_name='lstm_bayes_opt'
)

# Early Stopping Callback integrieren
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Hyperparameter-Suche starten (Training und Validierung)
tuner.search(
    x={"timeseries_input": X_train_ts, "product_group_input": X_train_prod},
    y=y_train,
    epochs=70,
    validation_data=({"timeseries_input": X_val_ts, "product_group_input": X_val_prod}, y_val),
    callbacks=[early_stopping]
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Beste Hyperparameter:")
print(f"Sequence Length: {best_hp.get('sequence_length')}")
print(f"LSTM-Einheiten: {best_hp.get('lstm_units')}")
print(f"Dropoutrate: {best_hp.get('dropout_rate')}")
print(f"Embedding-Größe: {best_hp.get('embedding_size')}")
print(f"Lernrate: {best_hp.get('learning_rate')}")

# Bestes Modell erstellen und anzeigen
model = tuner.hypermodel.build(best_hp)
model.summary()

# Finaler Test auf dem Testdatensatz
test_loss, test_mae, test_smape, test_r2 = model.evaluate(
    x={"timeseries_input": X_test_ts, "product_group_input": X_test_prod},
    y=y_test
)

# Fehler auf Originalmaßstab umrechnen
range_val = scaler.data_max_[2] - scaler.data_min_[2]
mae_unscaled = test_mae * range_val
loss_unscaled = test_loss * (range_val ** 2)

print(f"Test Loss: {loss_unscaled}, Test MAE: {mae_unscaled}, Test sMAPE: {test_smape}, Test R²: {test_r2}")

model.save('final_model.keras')
