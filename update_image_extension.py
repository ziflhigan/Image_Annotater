import json
import os
from pathlib import Path
from typing import List


def find_schema_files(root_folder: str) -> List[Path]:
    """
    Find all JSON schema files in all schema folders within the root folder.
    Looks for folders with 'schema_' prefix and recursively searches for JSON files.
    """
    schema_files = []
    root_path = Path(root_folder)

    if not root_path.exists():
        print(f"Root folder {root_folder} does not exist!")
        return schema_files

    # Find all folders that start with "schema_"
    for item in root_path.iterdir():
        if item.is_dir() and item.name.startswith("schema_"):
            print(f"Found schema folder: {item}")
            # Recursively search for JSON files in this folder
            for json_file in item.rglob("*.json"):
                schema_files.append(json_file)
                print(f"  Found schema file: {json_file}")

    return schema_files


def update_image_extension(image_path: str) -> tuple[str, bool]:
    """
    Update image path extension to .jpg if it's not already .jpg
    Returns: (updated_path, was_changed)
    """
    path = Path(image_path)

    # Check if extension is already .jpg
    if path.suffix.lower() == '.jpg':
        return image_path, False

    # Change extension to .jpg
    new_path = path.with_suffix('.jpg')
    return str(new_path), True


def main():
    """
    Main function to update image extensions in all schema files.
    """
    root_folder = "annotated_dataset"

    # Find all schema files across all schema folders
    schema_files = find_schema_files(root_folder)

    if not schema_files:
        print("No schema files found!")
        return

    total_files = len(schema_files)
    files_updated = 0
    total_changes = 0

    # Process each schema file
    for schema_file in schema_files:
        try:
            # Load the schema
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)

            # Get current image path
            current_path = schema_data.get('image_path', '')

            if not current_path:
                print(f"Warning: No image_path found in {schema_file.name}")
                continue

            # Update extension if needed
            new_path, was_changed = update_image_extension(current_path)

            if was_changed:
                # Update the schema data
                schema_data['image_path'] = new_path

                # Save the updated schema back to file
                with open(schema_file, 'w', encoding='utf-8') as f:
                    json.dump(schema_data, f, indent=2, ensure_ascii=False)

                print(f"Updated {schema_file.name}:")
                print(f"  From: {current_path}")
                print(f"  To:   {new_path}")

                files_updated += 1
                total_changes += 1
            else:
                print(f"No change needed for {schema_file.name} (already .jpg)")

        except Exception as e:
            print(f"Error processing {schema_file}: {e}")

    # Print summary
    print(f"\nSummary:")
    print(f"Total schema files found: {total_files}")
    print(f"Files updated: {files_updated}")
    print(f"Files already with .jpg extension: {total_files - files_updated}")
    print(f"Total path changes made: {total_changes}")

    if files_updated > 0:
        print(f"\n✅ Successfully updated {files_updated} schema files!")
    else:
        print(f"\nℹ️  All schema files already had .jpg extensions.")


if __name__ == "__main__":
    main()