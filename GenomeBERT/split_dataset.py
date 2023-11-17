from sklearn.model_selection import train_test_split




train_set, test_val_set = train_test_split(rawdata, test_size=0.3, random_state=42)
val_set, test_set = train_test_split(test_val_set, test_size=0.5, random_state=42)