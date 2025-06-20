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


def get_expected_image_path(schema_file_path: Path, image_filename: str) -> str:
    """
    Construct the expected image path based on the schema file's location.

    Example:
    - Schema file: annotated_dataset/schema_Food/Western/abc.json
    - Image filename: abc.jpg
    - Expected path: dataset/Food/Western/abc.jpg
    """
    # Get the parts of the schema path
    parts = schema_file_path.parts

    # Find the schema folder (starts with "schema_")
    schema_folder_idx = None
    category = None

    for i, part in enumerate(parts):
        if part.startswith("schema_"):
            schema_folder_idx = i
            # Extract category from schema folder name (e.g., "schema_Food" -> "Food")
            category = part[7:]  # Remove "schema_" prefix
            break

    if schema_folder_idx is None or category is None:
        return None

    # Get the sub-path after the schema folder
    sub_path_parts = parts[schema_folder_idx + 1:-1]  # Exclude the filename itself

    # Construct the expected image path
    path_parts = ["dataset", category] + list(sub_path_parts) + [image_filename]
    expected_path = str(Path(*path_parts))

    return expected_path


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


def get_expected_image_path(schema_file_path: Path, image_filename: str) -> str:
    """
    Construct the expected image path based on the schema file's location.

    Example:
    - Schema file: annotated_dataset/schema_Food/Western/abc.json
    - Image filename: abc.jpg
    - Expected path: dataset/Food/Western/abc.jpg
    """
    # Get the parts of the schema path
    parts = schema_file_path.parts

    # Find the schema folder (starts with "schema_")
    schema_folder_idx = None
    category = None

    for i, part in enumerate(parts):
        if part.startswith("schema_"):
            schema_folder_idx = i
            # Extract category from schema folder name (e.g., "schema_Food" -> "Food")
            category = part[7:]  # Remove "schema_" prefix
            break

    if schema_folder_idx is None or category is None:
        return None

    # Get the sub-path after the schema folder
    sub_path_parts = parts[schema_folder_idx + 1:-1]  # Exclude the filename itself

    # Construct the expected image path
    path_parts = ["dataset", category] + list(sub_path_parts) + [image_filename]
    expected_path = str(Path(*path_parts))

    return expected_path


def fix_image_path(schema_file_path: Path, current_image_path: str) -> tuple[str, bool]:
    """
    Fix the image path if it's missing sub-folders or has wrong structure.
    Returns: (corrected_path, was_changed)
    """
    if not current_image_path:
        return current_image_path, False

    # Extract just the filename from current path
    current_path = Path(current_image_path)
    image_filename = current_path.name

    # Get the expected path based on schema location
    expected_path = get_expected_image_path(schema_file_path, image_filename)

    if expected_path is None:
        return current_image_path, False

    # Normalize paths for comparison (handle different separators)
    current_normalized = str(current_path).replace('\\', '/').replace('//', '/')
    expected_normalized = expected_path.replace('\\', '/').replace('//', '/')

    if current_normalized != expected_normalized:
        return expected_path, True

    return current_image_path, False


def main():
    """
    Main function to fix image paths and update extensions in all schema files.
    """
    root_folder = "annotated_dataset"

    # Find all schema files across all schema folders
    schema_files = find_schema_files(root_folder)

    if not schema_files:
        print("No schema files found!")
        return

    total_files = len(schema_files)
    files_updated = 0
    path_fixes = 0
    extension_fixes = 0

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

            original_path = current_path
            file_was_updated = False
            changes_made = []

            # Step 1: Fix path structure (missing folders)
            fixed_path, path_changed = fix_image_path(schema_file, current_path)
            if path_changed:
                current_path = fixed_path
                file_was_updated = True
                path_fixes += 1
                changes_made.append("path structure")

            # Step 2: Update extension to .jpg
            final_path, ext_changed = update_image_extension(current_path)
            if ext_changed:
                current_path = final_path
                file_was_updated = True
                extension_fixes += 1
                changes_made.append("extension")

            # Save changes if any were made
            if file_was_updated:
                schema_data['image_path'] = current_path

                # Save the updated schema back to file
                with open(schema_file, 'w', encoding='utf-8') as f:
                    json.dump(schema_data, f, indent=2, ensure_ascii=False)

                print(f"Updated {schema_file.name} ({', '.join(changes_made)}):")
                print(f"  From: {original_path}")
                print(f"  To:   {current_path}")

                files_updated += 1
            else:
                print(f"No changes needed for {schema_file.name}")

        except Exception as e:
            print(f"Error processing {schema_file}: {e}")

    # Print summary
    print(f"\nSummary:")
    print(f"Total schema files found: {total_files}")
    print(f"Files updated: {files_updated}")
    print(f"Path structure fixes: {path_fixes}")
    print(f"Extension fixes: {extension_fixes}")
    print(f"Files already correct: {total_files - files_updated}")

    if files_updated > 0:
        print(f"\n✅ Successfully updated {files_updated} schema files!")
    else:
        print(f"\nℹ️  All schema files already had correct paths and extensions.")


if __name__ == "__main__":
    main()