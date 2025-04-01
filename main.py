import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import ast

data = pd.read_csv("Arrest_Data_from_2010_to_2019.csv")
data.head()

nrows = len(data)
print("Number of rows:", nrows)

data['Arrest Date'] = pd.to_datetime(data['Arrest Date'], format='%m/%d/%Y')
data_2018 = data[data['Arrest Date'].dt.year == 2018]
data_2018

nrows_2018 = len(data_2018)
print("Number of rows corresponding to 2018 bookings:", nrows_2018)

area_arrest_counts = data_2018.groupby('Area Name').size().reset_index(name='Arrest Count')
most_arrests_area = area_arrest_counts.loc[area_arrest_counts['Arrest Count'].idxmax()]

most_arrests_area_data = data_2018[data_2018['Area Name'] == most_arrests_area['Area Name']]
most_arrests_count = len(most_arrests_area_data)

print(f"The area with most arrests in 2018 is {most_arrests_area['Area Name']}.")
print(f"Number of arrests: {most_arrests_count}")

charge_groups = ['Vehicle Theft', 'Robbery', 'Burglary', 'Receive Stolen Property']
filtered_data = data[(data['Arrest Date'].dt.year == 2018) & (data['Charge Group Description'].isin(charge_groups))]
quantile_95 = filtered_data['Age'].quantile(0.95)

print("95% quantlie of age of arrestee in 2018:", quantile_95)


minor_groups = ['Pre-Delinquency', 'Non-Criminal Detention']
filtered_data = data[
    (data['Arrest Date'].dt.year == 2018) &
    (~data['Charge Group Description'].isin(minor_groups)) &
    (data['Charge Group Description'].notnull())
]

mean_age = filtered_data['Age'].mean()
std_age = filtered_data['Age'].std()
grouped_age = filtered_data.groupby('Charge Group Description')['Age'].mean()

z_scores = (grouped_age - mean_age) / std_age
max_z = z_scores.abs().max()

print("Largest absolute Z-score:", max_z)


bradbury_coords = (34.050536, -118.247861)
radius = 6371

filtered_data = data[
    (data['Arrest Date'].dt.year == 2018) &
    (data['Location'] != (0, 0))
]

def calculate_distance(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2-lon1)
    distance = radius * np.arccos(np.sin(phi1)*np.sin(phi2)+np.cos(phi1)*np.cos(phi2)*np.cos(delta_lambda))
    return distance

count_2km = 0
for index, row in filtered_data.iterrows():
    arrest_coords = eval(row['Location'])
    distance = calculate_distance(bradbury_coords[0], bradbury_coords[1], arrest_coords[0], arrest_coords[1])
    if distance <= 2:
        count_2km += 1

print("Number of arrest incidents within 2 km of the Bradbury Building in 2018:", count_2km)


pico_data = data[
    (data['Arrest Date'].dt.year == 2018) & 
    (data['Address'].str.contains("Pico", case=False))
].copy()

pico_data['Coordinates'] = pico_data['Location'].apply(ast.literal_eval)
pico_data.loc[:, 'Latitude'] = pico_data['Coordinates'].apply(lambda x: float(x[0]))
pico_data.loc[:, 'Longitude'] = pico_data['Coordinates'].apply(lambda x: float(x[1]))

lat_mean = pico_data['Latitude'].mean()
lat_std = pico_data['Latitude'].std()
lon_mean = pico_data['Longitude'].mean()
lon_std = pico_data['Longitude'].std()

filtered_pico_data = pico_data[
    (pico_data['Latitude'] >= lat_mean - 2*lat_std) & 
    (pico_data['Latitude'] <= lat_mean + 2*lat_std) &
    (pico_data['Longitude'] >= lon_mean - 2*lon_std) & 
    (pico_data['Longitude'] <= lon_mean + 2*lon_std)
]

west = filtered_pico_data['Longitude'].min()
east = filtered_pico_data['Longitude'].max()
length_pico = abs(west - east)*111.32  

num_incidents = len(filtered_pico_data)
incidents_per_km = num_incidents / length_pico if length_pico > 0 else 0

print("Number of arrest incidents per kilometer on Pico Boulevard in 2018:", incidents_per_km)

filtered_data = data[
    (data['Arrest Date'] < '2019-01-01') &
    (data['Charge Group Code'].notnull()) &
    (data['Charge Group Code'] != 99)
]

total_arrests = len(filtered_data)
charge_counts = filtered_data['Charge Group Code'].value_counts()

area_charge_counts = filtered_data.groupby(['Area ID', 'Charge Group Code']).size().reset_index(name='Area Count')
area_totals = filtered_data.groupby('Area ID').size().reset_index(name='Total Area Count')
area_data = pd.merge(area_charge_counts, area_totals, on='Area ID')

area_data['City Probability'] = area_data['Charge Group Code'].map(charge_counts/total_arrests)
area_data['Area Probability'] = area_data['Area Count'] / area_data['Total Area Count']
area_data['Ratio'] = area_data['Area Probability'] / area_data['City Probability']

top_ratios = area_data.nlargest(5, 'Ratio')
avg_top_ratios = top_ratios['Ratio'].mean()

print("Average:", avg_top_ratios)

felony_data = data[
    (data['Arrest Date'].dt.year >= 2010) &
    (data['Arrest Date'].dt.year <= 2018) &
    (data['Arrest Type Code'] == 'F') 
]
felony_counts = felony_data.groupby(felony_data['Arrest Date'].dt.year).size().reset_index(name='Count')

X = felony_counts['Arrest Date'].values.reshape(-1, 1) 
y = felony_counts['Count'].values 
model = LinearRegression()
model.fit(X, y)

predicted_2019 = model.predict(np.array([[2019]]))
print("Projected number:", predicted_2019)

 
# Rounded to the nearest integer results in 31038



