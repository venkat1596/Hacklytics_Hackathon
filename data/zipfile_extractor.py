import os
import zipfile
import shutil
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import time


def extract_zip(zip_path, extract_to=None, password=None, verbose=True):
    """
    Extract contents of a zip file to a specified directory with detailed progress reporting.

    Args:
        zip_path (str): Path to the zip file
        extract_to (str, optional): Directory to extract to. If None, extracts to a folder with the zip name
        password (str, optional): Password for encrypted zip files (encoded as bytes if provided)
        verbose (bool): Whether to show progress and information

    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    # Convert the zip path to a Path object for easier manipulation
    zip_path = Path(zip_path)

    # If no extraction path provided, create a folder with the zip name
    if extract_to is None:
        extract_to = zip_path.with_suffix('')

    # Ensure the extraction directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Convert password to bytes if provided
    pwd_bytes = password.encode() if password else None

    try:
        # Open the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files and total size for progress reporting
            file_list = zip_ref.infolist()
            total_size = sum(file_info.file_size for file_info in file_list)
            total_files = len(file_list)

            if verbose:
                print(f"📦 Extracting: {zip_path.name}")
                print(f"📊 Summary: {total_files} files, {total_size / 1024 / 1024:.2f} MB")
                print(f"📂 Destination: {extract_to}")
                print("-" * 60)

            # Check if the zip is password-protected and no password was provided
            if any(file_info.flag_bits & 0x1 for file_info in file_list) and not password:
                print("🔒 Error: This ZIP file is password-protected. Please provide a password.")
                return False

            # Create two progress bars - one for overall progress, one for current file
            extracted_size = 0
            extracted_files = 0

            # Main progress bar for overall extraction
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                      desc="Overall progress", disable=not verbose,
                      bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as overall_pbar:

                # Second progress bar for file count
                with tqdm(total=total_files, unit='files', disable=not verbose,
                          desc="Files extracted", position=1, leave=True,
                          bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} files') as file_pbar:

                    # Track time for rate limiting updates
                    last_update = time.time()
                    current_file = ""

                    for file_info in file_list:
                        try:
                            # Update current file display (not too frequently to avoid flickering)
                            if time.time() - last_update > 0.1:  # Update display every 100ms
                                current_filename = file_info.filename
                                # Truncate long filenames for display
                                if len(current_filename) > 40:
                                    current_filename = "..." + current_filename[-37:]

                                if verbose and current_file != current_filename:
                                    # Clear the line and write the current file
                                    current_file = current_filename
                                    # Position the cursor below the progress bars
                                    tqdm.write(f"🔄 Extracting: {current_file}")
                                last_update = time.time()

                            # Extract the file
                            zip_ref.extract(file_info, path=extract_to, pwd=pwd_bytes)

                            # Update progress bars
                            extracted_size += file_info.file_size
                            overall_pbar.update(file_info.file_size)

                            extracted_files += 1
                            file_pbar.update(1)

                        except zipfile.BadZipFile:
                            tqdm.write(f"❌ Error: Corrupted file in archive: {file_info.filename}")
                        except RuntimeError as e:
                            if "password required" in str(e).lower() or "bad password" in str(e).lower():
                                tqdm.write(f"🔑 Error: Password incorrect or required for: {file_info.filename}")
                            else:
                                tqdm.write(f"⚠️ Error extracting {file_info.filename}: {e}")
                        except Exception as e:
                            tqdm.write(f"❓ Unexpected error extracting {file_info.filename}: {e}")

            if verbose:
                print("-" * 60)
                print(
                    f"✅ Extraction complete: {extracted_files}/{total_files} files, {extracted_size / 1024 / 1024:.2f} MB")
                if extracted_files < total_files:
                    print(f"⚠️ Warning: {total_files - extracted_files} files could not be extracted")

            return extracted_files > 0

    except zipfile.BadZipFile:
        print(f"❌ Error: The file '{zip_path}' is not a valid ZIP file or is corrupted.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{zip_path}' was not found.")
    except PermissionError:
        print(f"❌ Error: Permission denied when accessing '{zip_path}'.")
    except Exception as e:
        print(f"❓ Unexpected error: {e}")

    return False


def main():
    """Command line interface for the zip extractor"""
    parser = argparse.ArgumentParser(description='Extract ZIP files with detailed progress reporting.')
    parser.add_argument('-o', '--output-dir', help='Directory to extract files to (default: folder with zip name)')
    parser.add_argument('-p', '--password', help='Password for encrypted ZIP files')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress progress information')

    args = parser.parse_args()

    zip_path = Path("./Jpegs.zip").absolute().__str__()

    success = extract_zip(
        zip_path,
        extract_to=args.output_dir,
        password=args.password,
        verbose=not args.quiet
    )

    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()