import pandas as pd

# Load the dataset
file_path = r"D:\Placement Prediction\Data\MR_dataset.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Define a function to capitalize each word and remove extra spaces at the end
def clean_string(cell_value):
    if isinstance(cell_value, str):
        # Capitalize each word
        cell_value = ' '.join(word.capitalize() for word in cell_value.split())
        # Remove extra spaces at the end
        cell_value = cell_value.strip()
    return cell_value


# Apply the function to all cells in the DataFrame
df = df.applymap(clean_string)

# Remove rows with null or missing values
df = df.dropna()

# Define the file path for the new CSV file
new_csv_file_path = r"D:\Placement Prediction\Data\cleaned_data_cleaned.csv"

# Save the modified DataFrame to a new CSV file
df.to_csv(new_csv_file_path, index=False)

print(f"Modified data has been saved to {new_csv_file_path}")
