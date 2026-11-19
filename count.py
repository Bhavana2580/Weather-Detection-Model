import os

dataset_path = "Dataset"

folders = os.listdir(dataset_path)

print("Folders found:")
print(folders)
print()

for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(folder, len(images))