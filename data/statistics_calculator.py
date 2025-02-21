import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class NIFTIStatisticsAnalyzer:
    """
    A class to analyze NIFTI images and calculate various statistics.
    This class handles both individual image statistics and aggregate dataset statistics.
    """

    def __init__(self, input_dir):
        """
        Initialize the analyzer with input directory and prepare data structures.

        Args:
            input_dir (str): Directory containing .nii.gz files
        """
        self.input_dir = Path(input_dir)
        self.statistics = {
            'dataset_summary': {
                'total_images': 0,
                'scanner_distribution': defaultdict(int),
                'aggregate_statistics': {
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'median': None,
                    'total_voxels': 0
                }
            },
            'individual_images': {},
            'metadata': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_directory': str(input_dir)
            }
        }

    def identify_scanner_strength(self, filename):
        """
        Determine the scanner strength from filename.

        Args:
            filename (str): Name of the NIFTI file

        Returns:
            str: Scanner strength ('1.5T' or '3T')
        """
        filename = filename.upper()
        if 'HH' in filename:
            return '3T'
        elif any(id in filename for id in ['GUYS', 'IOP']):
            return '1.5T'
        return 'Unknown'

    def calculate_image_statistics(self, nifti_data):
        """
        Calculate statistics for a single image.

        Args:
            nifti_data (numpy.ndarray): Image data array

        Returns:
            dict: Dictionary containing calculated statistics
        """
        return {
            'shape': list(nifti_data.shape),
            'dimensions': len(nifti_data.shape),
            'total_voxels': nifti_data.size,
            'statistics': {
                'mean': float(np.mean(nifti_data)),
                'std': float(np.std(nifti_data)),
                'min': float(np.min(nifti_data)),
                'max': float(np.max(nifti_data)),
                'median': float(np.median(nifti_data)),
                'non_zero_voxels': int(np.count_nonzero(nifti_data)),
                'zero_voxels': int(nifti_data.size - np.count_nonzero(nifti_data))
            }
        }

    def update_dataset_statistics(self, image_stats, scanner_strength):
        """
        Update the aggregate statistics for the entire dataset.

        Args:
            image_stats (dict): Statistics of individual image
            scanner_strength (str): Scanner strength of the image
        """
        self.statistics['dataset_summary']['total_images'] += 1
        self.statistics['dataset_summary']['scanner_distribution'][scanner_strength] += 1

        # Update aggregate statistics
        agg_stats = self.statistics['dataset_summary']['aggregate_statistics']
        if agg_stats['mean'] is None:
            # First image
            for key in ['mean', 'std', 'min', 'max', 'median']:
                agg_stats[key] = image_stats['statistics'][key]
        else:
            # Update running statistics
            n = self.statistics['dataset_summary']['total_images']
            prev_mean = agg_stats['mean']
            curr_mean = image_stats['statistics']['mean']

            # Update mean
            agg_stats['mean'] = ((n - 1) * prev_mean + curr_mean) / n

            # Update min/max
            agg_stats['min'] = min(agg_stats['min'], image_stats['statistics']['min'])
            agg_stats['max'] = max(agg_stats['max'], image_stats['statistics']['max'])

        agg_stats['total_voxels'] += image_stats['total_voxels']

    def analyze_images(self):
        """
        Analyze all NIFTI images in the input directory and calculate statistics.
        """
        print("Starting NIFTI image analysis...")

        for file_path in self.input_dir.glob('*.nii.gz'):
            print(f"Processing: {file_path.name}")

            try:
                # Load NIFTI image
                img = nib.load(str(file_path))
                data = img.get_fdata()

                # Calculate statistics for this image
                scanner_strength = self.identify_scanner_strength(file_path.name)
                image_stats = self.calculate_image_statistics(data)

                # Add additional metadata
                image_stats['file_info'] = {
                    'filename': file_path.name,
                    'scanner_strength': scanner_strength,
                    'file_size': os.path.getsize(file_path),
                    'affine_matrix': img.affine.tolist()
                }

                # Store individual image statistics
                self.statistics['individual_images'][file_path.name] = image_stats

                # Update dataset-wide statistics
                self.update_dataset_statistics(image_stats, scanner_strength)

            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
                continue

    def save_statistics(self, output_file):
        """
        Save all calculated statistics to a JSON file.

        Args:
            output_file (str): Path to output JSON file
        """
        # Convert defaultdict to regular dict for JSON serialization
        self.statistics['dataset_summary']['scanner_distribution'] = \
            dict(self.statistics['dataset_summary']['scanner_distribution'])

        with open(output_file, 'w') as f:
            json.dump(self.statistics, f, indent=4)

        print(f"Statistics saved to: {output_file}")


def main():
    """
    Main function to run the NIFTI image analysis.
    """
    # Set your input directory containing .nii.gz files
    input_directory = "./T1-organized/source_1.5T"

    # Create analyzer instance
    analyzer = NIFTIStatisticsAnalyzer(input_directory)

    # Run analysis
    analyzer.analyze_images()

    # Save results
    output_file = os.path.join(input_directory, "nifti_statistics.json")
    analyzer.save_statistics(output_file)

    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total images processed: {analyzer.statistics['dataset_summary']['total_images']}")
    print("Scanner distribution:")
    for scanner, count in analyzer.statistics['dataset_summary']['scanner_distribution'].items():
        print(f"  {scanner}: {count} images")


if __name__ == "__main__":
    main()