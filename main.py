import csv
from Customer import Customer


def read_data():
    with open('training_data.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            customers.append(Customer(row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]))

customers = []
read_data()


