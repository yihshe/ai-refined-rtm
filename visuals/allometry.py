# Analyze the allometric database and plot the data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the paths to read the data and save the plots
DATA_PATH = '/maps/ys611/ai-refined-rtm/data/allometry/Data.csv'
SAVE_PATH = '/maps/ys611/ai-refined-rtm/visuals/allometry/'

# TODO test with different biomes and zones
# Load the data
df = pd.read_csv(DATA_PATH)  # Replace with your CSV file path

# Define the biogeographic zones of interest
zones_of_interest = ['Zone1', 'Zone2', 'Zone3']  # Replace with your zones

# Filter the data by biogeographic zones
filtered_df = df[df['Biogeographic_zone'].isin(zones_of_interest)]

# Plot H (Height) vs CD (Crown Diameter)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CD', y='H', data=filtered_df)
plt.title('Scatter Plot of Tree Height vs Crown Diameter')
plt.xlabel('Crown Diameter (CD)')
plt.ylabel('Height (H)')
plt.show()

# Further analysis to decide the best curve fit can be based on the plot
