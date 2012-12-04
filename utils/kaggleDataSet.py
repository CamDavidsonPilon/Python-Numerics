#Author: John Ramney

import requests

# The direct link to the Kaggle data set
data_url = 'http://www.kaggle.com/c/digit-recognizer/download/train.csv'

# The local path where the data set is saved.
local_filename = "train.csv"

# Kaggle Username and Password
kaggle_info = {'UserName': "my_username", 'Password': "my_password"}

# Attempts to download the CSV file. Gets rejected because we are not logged in.
r = requests.get(data_url)

# Login to Kaggle and retrieve the data.
r = requests.post(r.url, data = kaggle_info, prefetch = False)

# Writes the data to a local file one chunk at a time.
f = open(local_filename, 'w')
for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
    if chunk: # filter out keep-alive new chunks
        f.write(chunk)
f.close()
