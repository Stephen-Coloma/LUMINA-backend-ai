from pathlib import Path

def rename_pet_files(dataset_path: Path):
    for patient_folder in dataset_path.iterdir():
        pet_folder = patient_folder / 'PET'
        if not pet_folder.exists() or not pet_folder.is_dir():
            continue

        # Determine max number from files already starting with '1-'
        files = sorted(pet_folder.glob('1-*.dcm'))
        if files:
            max_number = int(files[-1].stem.split('-')[1])
        else:
            max_number = 0

        # obtain files that needs to be renamed
        files_to_rename = [
            f for f in pet_folder.glob('*.dcm')
            if not f.stem.startswith('1-') or not f.stem.split('-')[1].isdigit()
        ]

        # Rename only files not starting with '1-'
        counter = max_number + 1
        for file in files_to_rename:
            new_name = f"1-{counter:03d}{file.suffix}"
            new_path = file.with_name(new_name)
            file.rename(new_path)
            counter += 1