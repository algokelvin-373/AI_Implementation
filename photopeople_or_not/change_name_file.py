import os

def main():
    folder_path = r"D:\AlgoKelvin\My_Project\Juni 2025\AI_Implementation\photopeople_or_not\dataset\val\non-human"
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
    else:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        files.sort()

        for index, filename in enumerate(files, start=1):
            name, ext = os.path.splitext(filename)
            new_name = f"{index}{ext}"

            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

        print(f"Done! {len(files)} file is changed.")


if __name__ == "__main__":
    main()