import tensorflow as tf
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# age groups: <40, 40-50, 50-60, 60-70, >70


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Response')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds



CSV_COLUMN_NAMES = ['Age', 'DrivingLicense', 'Region', 'PreviouslyInsured', 'VehicleAge', 'VehicleDamage',
                    'AnnualPremium', 'PolicySalesChannel', 'Vintage', 'Response']
RESPONSES = ['Interested', 'Not Interested']

dataframe = pd.read_csv("training_data.csv")
tf.keras.backend.set_floatx('float64')
dataframe.drop(columns='id')
train, test = train_test_split(dataframe, test_size=0.15)
train, val = train_test_split(train, test_size=0.15)


feature_columns = []

# numeric cols
for header in ['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age = feature_column.numeric_column('Age')
age_buckets = feature_column.bucketized_column(age, boundaries=[40, 50, 60, 70])
feature_columns.append(age_buckets)

# indicator cols
indicator_column_names = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
for col_name in indicator_column_names:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(
        col_name, dataframe[col_name].unique())
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(3, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)



