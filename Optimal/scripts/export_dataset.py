# scripts/export_dataset.py
import os
import json
import shutil
import argparse
import zipfile
from datetime import datetime

def create_export_package(output_dir="output", export_format="zip"):
    """Create an export package of the dataset."""
    # Create timestamp for the export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_name = f"multimodal_interaction_dataset_{timestamp}"
    
    if export_format == "zip":
        # Create a zip file
        zip_path = f"{export_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the output directory
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    # Skip large files or temporary files
                    if file.endswith(".tmp") or file.endswith(".log"):
                        continue
                    
                    file_path = os.path.join(root, file)
                    # Add file to zip with a path relative to the output directory
                    arcname = os.path.relpath(file_path, start=os.path.dirname(output_dir))
                    zipf.write(file_path, arcname=arcname)
        
        print(f"Created export package: {zip_path}")
        return zip_path
    
    elif export_format == "directory":
        # Create a directory
        export_dir = f"{export_name}"
        os.makedirs(export_dir, exist_ok=True)
        
        # Copy the output directory structure
        for root, dirs, files in os.walk(output_dir):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                rel_path = os.path.relpath(dir_path, start=output_dir)
                os.makedirs(os.path.join(export_dir, rel_path), exist_ok=True)
            
            for file in files:
                # Skip large files or temporary files
                if file.endswith(".tmp") or file.endswith(".log"):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, start=output_dir)
                shutil.copy2(file_path, os.path.join(export_dir, rel_path))
        
        print(f"Created export directory: {export_dir}")
        return export_dir
    
    else:
        print(f"Unsupported export format: {export_format}")
        return None

def main():
    """Main function to create an export package."""
    parser = argparse.ArgumentParser(description="Export multimodal interaction dataset")
    parser.add_argument("--format", choices=["zip", "directory"], default="zip", help="Export format")
    parser.add_argument("--output", default="output", help="Output directory to export")
    args = parser.parse_args()
    
    # Check if output directory exists
    if not os.path.exists(args.output):
        print(f"Error: Output directory not found: {args.output}")
        return
    
    # Create export package
    export_path = create_export_package(output_dir=args.output, export_format=args.format)
    
    if export_path:
        print(f"Export complete: {export_path}")
    else:
        print("Export failed")

if __name__ == "__main__":
    main()