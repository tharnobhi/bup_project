import pandas as pd

# Load the PCA data
data = pd.read_csv("crossCheckResult/mfcc_data_clusters_k3.csv")

# Convert time column to HH:MM:SS format
data['time'] = data['time'].str.replace('_', ':').str[:-4]

# Filter by date range
while True:
    start_date = input("Enter start date (YYYY-MM-DD): ")
    try:
        pd.Timestamp(start_date)
        break
    except ValueError:
        print("Invalid date format. Please enter a valid date in YYYY-MM-DD format.")

while True:
    end_date = input("Enter end date (YYYY-MM-DD): ")
    try:
        pd.Timestamp(end_date)
        if end_date >= start_date:
            break
        else:
            print("End date must be later than start date.")
    except ValueError:
        print("Invalid date format. Please enter a valid date in YYYY-MM-DD format.")

filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# Filter by time range
while True:
    start_time = input("Enter start time (HH:MM:SS): ")
    try:
        pd.Timestamp(start_date + ' ' + start_time)
        break
    except ValueError:
        print("Invalid time format. Please enter a valid time in HH:MM:SS format.")

while True:
    end_time = input("Enter end time (HH:MM:SS): ")
    try:
        pd.Timestamp(end_date + ' ' + end_time)
        if end_date + ' ' + end_time >= start_date + ' ' + start_time:
            break
        else:
            print("End time must be later than start time.")
    except ValueError:
        print("Invalid time format. Please enter a valid time in HH:MM:SS format.")

filtered_data = filtered_data[(filtered_data['time'] >= start_time) & (filtered_data['time'] <= end_time)]
filtered_data = filtered_data.drop(columns=['label'])
filtered_data.to_csv("filtered_pca_data.csv", index=False)

# Print the filtered data
print(filtered_data)
