# Benefits of Data Wrangling

data = {"raw_data": [None, 23, 45, None, 67, "NA", 89]}
cleaned_data = [x for x in data["raw_data"] if x not in [None, "NA"]]

print("Raw Data:", data["raw_data"])
print("Cleaned Data:", cleaned_data)


# Cleaning Data

import pandas as pd

data = {
    'Name': ['Alice', 'Bob', None, 'David'],
    'Age': [25, None, 30, 'NA']
}

df = pd.DataFrame(data)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df_cleaned = df.dropna()

print("Original Data:")
print(df)

print("\nCleaned Data:")
print(df_cleaned)


# Normalizing Data

import numpy as np

data = np.array([10, 20, 30, 40, 50])
normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

print("Normalized Data:", normalized_data)


# Python Basics

print("Hello, World!")

x = 10
y = 20

print("Sum:", x + y)


# Reading and Writing CSV

import pandas as pd

data = {
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
}

df = pd.DataFrame(data)

df.to_csv('example.csv', index=False)

df_read = pd.read_csv('example.csv')
print(df_read)


# Reading JSON

import json

data = '{"Name": "Alice", "Age": 25}'

parsed_data = json.loads(data)

print(parsed_data)


# Parsing XML

import xml.etree.ElementTree as ET

xml_data = '<person><name>Alice</name><age>25</age></person>'

root = ET.fromstring(xml_data)

print("Name:", root.find('name').text)
print("Age:", root.find('age').text)


# Retrieving Database Schema Using SQLite

import sqlite3

conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

cursor.execute('CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)')
cursor.execute('INSERT INTO users VALUES (1, "Alice", 25)')
conn.commit()

cursor.execute("PRAGMA table_info(users)")
schema = cursor.fetchall()

print("Database Schema:")

for column in schema:
    print(column)
