import os
import urllib.request
def download_dataset():
    # URL to the raw CSV on GitHub (pooja2512 repository)
    url = "https://raw.githubusercontent.com/pooja2512/Adult-Census-Income/master/adult.csv"
    
    # Destination paths
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    dest_path = os.path.join(data_dir, "adult.csv")
    
    # Create directories if they do not exist
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Dataset successfully downloaded and saved to: {os.path.abspath(dest_path)}")
        
        # Verify download size and presence
        if os.path.exists(dest_path):
            file_size = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"File size: {file_size:.2f} MB")
            
            # Print first 3 lines to check format
            with open(dest_path, 'r') as f:
                print("First 3 lines of the file:")
                for _ in range(3):
                    print("  ", f.readline().strip())
        else:
            raise FileNotFoundError("Downloaded file not found on disk.")
            
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        print("Attempting fallback download from UCI repository...")
        try:
            # Fallback UCI url
            fallback_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            columns = ["age", "workclass", "fnlwgt", "education", "education.num", 
                       "marital.status", "occupation", "relationship", "race", "sex", 
                       "capital.gain", "capital-loss", "hours.per.week", "native.country", "income"]
            # We can download it using pandas if we install it, or download the raw data and add headers manually
            temp_path = os.path.join(data_dir, "adult_raw.data")
            urllib.request.urlretrieve(fallback_url, temp_path)
            
            # Construct CSV with header
            with open(temp_path, 'r') as raw_file, open(dest_path, 'w') as csv_file:
                csv_file.write(",".join(columns) + "\n")
                for line in raw_file:
                    if line.strip():
                        csv_file.write(line)
            
            print(f"Successfully constructed dataset from UCI fallback to: {os.path.abspath(dest_path)}")
            # Cleanup temp raw file
            os.remove(temp_path)
        except Exception as fallback_e:
            print(f"Fallback download also failed: {fallback_e}")
            raise
if __name__ == "__main__":
    download_dataset()
