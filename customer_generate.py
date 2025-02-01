import pandas as pd
import numpy as np

# Load the existing data
customers = pd.read_csv('C:/Users/aayus/Desktop/team-OK/data/customers.csv')
locations = pd.read_csv('C:/Users/aayus/Desktop/team-OK/data/mumbai_locations.csv')

# Ensure location_id is of type string
locations['location_id'] = locations['location_id'].astype(str)

# Define income thresholds
high_income_threshold = 100000
middle_income_threshold = 50000

# Define a function to assign location based on income
def assign_location(row):
    if row['income'] > high_income_threshold:
        return locations[locations['primary_demographic'] == 'High Income']['location_id'].sample().values[0]
    elif row['income'] > middle_income_threshold:
        return locations[locations['primary_demographic'] == 'Middle Income']['location_id'].sample().values[0]
    else:
        return locations[locations['primary_demographic'] == 'Lower Middle Income']['location_id'].sample().values[0]

# Apply the function to assign location_id
customers['location_id'] = customers.apply(assign_location, axis=1)

# Save the new DataFrame to a CSV file
customers.to_csv('C:/Users/aayus/Desktop/team-OK/data/new_customers.csv', index=False)