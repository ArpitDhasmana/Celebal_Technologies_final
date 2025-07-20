import pandas as pd
import os

def load_dataset(csv_path, image_folder, include_labels=True):
    df = pd.read_csv(csv_path)
    image_paths = []
    labels = [] if include_labels else None

    label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']

    for _, row in df.iterrows():
        image_name = row['image_id'] + ".jpg"
        image_path = os.path.join(image_folder, image_name)
        image_paths.append(image_path)

        if include_labels:
            # Convert one-hot encoding to label string
            label_index = row[label_cols].values.argmax()
            label = label_cols[label_index]
            labels.append(label)

    return image_paths, labels
