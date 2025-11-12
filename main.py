import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from google.colab import data_table

data_table.disable_dataframe_formatter()

# Ladda dataset
house_prices = pd.read_csv("/content/SwedenHousingPrices.csv")

# Display descriptive statistics BEFORE cleaning
print("Descriptive statistics BEFORE cleaning:")
display(house_prices.describe())


# Drop specified columns
house_prices = house_prices.drop(columns=['ad_id', 'date_published', 'coordenates'])

# Filter out rows where 'typology' is in the specified list
house_prices = house_prices[~house_prices['typology'].isin(['APARTMENT','AGRICULTURAL_ESTATE', 'LINKED_HOUSE', 'VACATION_HOUSE', 'WINTERIZED_VACATION_HOME', 'ESTATE_WITHOUT_CULTIVATION', 'FORESTING_ESTATE', 'HOMESTEAD', 'VACATION_HOME', 'TWIN_HOUSE', 'TERRACED_HOUSE', 'ROW_HOUSE', 'PLOT', 'OTHER'])]

# Omvandla kategoriska variabler: 'typology'
encoder_typology = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_typology = encoder_typology.fit_transform(house_prices[['typology']])
encoded_typology_df = pd.DataFrame(encoded_typology, columns=encoder_typology.get_feature_names_out(['typology']), index=house_prices.index)
house_prices = pd.concat([house_prices, encoded_typology_df], axis=1)

# Drop the original 'typology' column after one-hot encoding
house_prices = house_prices.drop(columns=['typology'])

# Rename columns as requested
house_prices = house_prices.rename(columns={
    'land_area_sqm': 'tomtyta',
    'living_area_sqm': 'boyta',
    'number_rooms': 'rum',
    'typology_HOUSE': 'hus',
    'asking_price_sek': 'utgångspris',
    'sqm_price_sek': 'pris_sqm'
})

# Skapa ny feature: 'municipality' från 'location'
if 'location' in house_prices.columns:
    # Extract municipality name from 'location'
    house_prices['municipality'] = house_prices['location'].apply(lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else 'Unknown')

    # Drop original 'address' and 'location' columns
    house_prices = house_prices.drop(columns=['address', 'location'])

# List of municipalities in Västra Götalands län (provided by user)
vastra_gotaland_municipalities = ['Ale kommun', 'Alingsås kommun', 'Bengtsfors kommun', 'Bollebygds kommun', 'Borås kommun', 'Dals-Eds kommun', 'Essunga kommun', 'Falköpings kommun', 'Färgelanda kommun', 'Grästorps kommun', 'Gullspångs kommun', 'Göteborgs kommun', 'Götene kommun', 'Herrljunga kommun', 'Hjo kommun', 'Härryda kommun', 'Karlsborgs kommun', 'Kungälvs kommun', 'Lerums kommun', 'Lidköpings kommun', 'Lilla Edets kommun', 'Lysekils kommun', 'Mariestads kommun', 'Mark kommun', 'Melleruds kommun', 'Mjölby kommun', 'Munkedals kommun', 'Mölndals kommun', 'Orust kommun', 'Partille kommun', 'Skara kommun', 'Skövde kommun', 'Sotenäs kommun', 'Stenungsunds kommun', 'Strömstads kommun', 'Svenljunga kommun', 'Tanum kommun', 'Tibro kommun', 'Tidaholms kommun', 'Tjörn kommun', 'Tranemo kommun', 'Trollhättans kommun', 'Töreboda kommun', 'Uddevalla kommun', 'Ulricehamns kommun', 'Vara kommun', 'Vårgårda kommun', 'Vänersborgs kommun', 'Åmåls kommun', 'Öckerö kommun'] # Added Mölndals kommun to the list. Added Mjölby based on the user's input which was not in Västra Götalands län, but assuming user intended to include it based on previous turn. Removed 'Unknown' from the filter list.

# Filter by municipalities in Västra Götalands län
house_prices = house_prices[house_prices['municipality'].isin(vastra_gotaland_municipalities)].copy()

# Sort the DataFrame by 'municipality'
house_prices = house_prices.sort_values(by='municipality').reset_index(drop=True)

# Remove rows where 'tomtyta' is missing or zero
house_prices = house_prices[house_prices['tomtyta'].notna() & (house_prices['tomtyta'] > 0)].copy()

# Remove rows where 'tomtyta' is less than 200
house_prices = house_prices[house_prices['tomtyta'] >= 200].copy()

# Remove rows where 'boyta' is missing or zero
house_prices = house_prices[(house_prices['boyta'].notna()) & (house_prices['boyta'] > 0)].copy()

# Remove rows where 'rum' is less than 1
house_prices = house_prices[house_prices['rum'] >= 1].copy()


# Handle remaining missing values after filtering
numeric_cols = house_prices.select_dtypes(include=np.number).columns
house_prices[numeric_cols] = house_prices[numeric_cols].fillna(house_prices[numeric_cols].mean())

# Remove outliers using IQR for numerical columns
for col in ['tomtyta', 'boyta', 'rum', 'utgångspris']:
    if col in house_prices.columns:
        Q1 = house_prices[col].quantile(0.25)
        Q3 = house_prices[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        house_prices = house_prices[(house_prices[col] >= lower_bound) & (house_prices[col] <= upper_bound)].copy()


# Convert 'rum' (formerly number_rooms) to integer after handling missing values
house_prices['rum'] = house_prices['rum'].round().astype(int)

# Round remaining numerical columns to 2 decimal place
cols_to_round = [col for col in numeric_cols if col != 'rum' and col in house_prices.columns] # Check if column still exists after outlier removal
house_prices[cols_to_round] = house_prices[cols_to_round].round(1)

# Create interaction feature: boyta * rum (formerly living_area_sqm * number_rooms)
house_prices['boyta_rum_interaktion'] = house_prices['boyta'] * house_prices['rum']

# One-hot encode the 'municipality' column
encoder_municipality = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_municipality = encoder_municipality.fit_transform(house_prices[['municipality']])
encoded_municipality_df = pd.DataFrame(encoded_municipality, columns=encoder_municipality.get_feature_names_out(['municipality']), index=house_prices.index)
house_prices = pd.concat([house_prices, encoded_municipality_df], axis=1)

# Drop the original 'municipality' column after one-hot encoding
house_prices = house_prices.drop(columns=['municipality'])


# Define features (X) and target variable (y)
# 'utgångspris' (formerly 'asking_price_sek') is the target variable
# Drop specified typology columns from features
columns_to_drop_features = ['utgångspris', 'pris_sqm'] # Removed typology columns from here as they are handled earlier
X = house_prices.drop(columns=columns_to_drop_features, errors='ignore') # Use errors='ignore' in case some columns were already dropped
y = house_prices['utgångspris']

# Select numerical columns for scaling (excluding target and already dropped)
# Need to re-select numerical columns after one-hot encoding municipality
numeric_features = X.select_dtypes(include=np.number).columns

# Apply StandardScaler() to numerical features if there are any
if not numeric_features.empty:
  scaler = StandardScaler()
  X[numeric_features] = scaler.fit_transform(X[numeric_features])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state for reproducibility

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
prediction = model.predict(X_test)

# Evaluate the model
score_mse = mean_squared_error(y_test, prediction)
score_r2 = r2_score(y_test, prediction)
score_mae = mean_absolute_error(y_test, prediction)


print(f"Mean Squared Error: {score_mse}")
print(f"R-squared: {score_r2}")
print(f"Mean Absolute Error: {score_mae}")

# Display descriptive statistics AFTER cleaning
print("\nDescriptive statistics AFTER cleaning:")
display(house_prices.describe())
display(X.head()) # Displaying head of scaled features for confirmation