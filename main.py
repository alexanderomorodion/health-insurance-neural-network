import csv
import tensorflow as tf
import pandas as pd
from Customer import Customer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
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

train_path = tf.keras.utils.get_file(
    "training_data.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
'''
