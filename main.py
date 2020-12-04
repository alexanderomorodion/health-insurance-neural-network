import csv
import tensorflow as tf
import pandas as pd
from Customer import Customer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from numpy import argmax


# age groups: <40, 40-50, 50-60, 60-70, >70
age_dict: {0: 20, 1: 40, 2: 50, 3: 60, 4: 70}
veh_age_dict: {0: "< 1 Year", 1: "1-2 Year", 2: "> 2 Years"}
customers = []
age_list = []
region_list = []
veh_age_list = []

def read_data():
    i = 0
    csvfile = pd.read_csv('training_data.csv')
    for index, row in csvfile.iterrows():
        if i != 0:
            if row['Age'] < 40:
                age = 20
                age_list.append(0)
            elif row['Age'] < 50:
                age = 40
                age_list.append(1)
            elif row['Age'] < 60:
                age = 50
                age_list.append(2)
            elif row['Age'] < 70:
                age = 60
                age_list.append(3)
            else:
                age = 70
                age_list.append(4)

            if row['Vehicle_Age'] == '< 1 Year':
                veh_age_list.append(0)
            elif row['Vehicle_Age'] == '1-2 Year':
                veh_age_list.append(1)
            else:
                veh_age_list.append(2)
            customers.append(
                Customer(row[1], age, row[3], int(row[4]), row[5], row[6], row[7], int(row[8]), int(row[9]), row[10], row[11]))
            region_list.append(int(row[4]))

        i += 1


def onehot_encode(data):
    # define example
    values = array(data)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Response')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


'''
read_data()


age = tf.constant(5, name="age")
region = tf.constant(52, name="region")
veh_age = tf.constant(3, name="veh_age")

age_matrix = tf.one_hot(
    age_list, age, on_value=1.0, off_value=0.0, axis=-1).numpy()

region_matrix = tf.one_hot(
    region_list, region, on_value=1.0, off_value=0.0, axis=-1).numpy()

veh_age_matrix = tf.one_hot(
    veh_age_list, veh_age, on_value=1.0, off_value=0.0, axis=-1).numpy()


for i in range(age_matrix.shape[0]):
    customers[i].age = age_matrix[i]
    customers[i].region = region_matrix[i]
    customers[i].vehicle_age = veh_age_matrix[i]

'''

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
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
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
'''
#test_path = tf.keras.utils.get_file(
#    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

#test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # One hidden layer of 30 nodes
    hidden_units=[30],
    # The model must choose between 2 classes.
    n_classes=2)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
    '''


