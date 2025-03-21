import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

def save_diagnostic_image(img_array, title, output_path):
    """Save a diagnostic image showing the array details"""
    plt.figure(figsize=(12, 8))
    
    if len(img_array.shape) == 3:
        # Show RGB channels separately
        plt.subplot(2, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        
        # Show R, G, B channels separately
        plt.subplot(2, 2, 2)
        plt.imshow(img_array[:,:,0], cmap='gray')
        plt.title("Red Channel")
        
        plt.subplot(2, 2, 3)
        plt.imshow(img_array[:,:,1], cmap='gray')
        plt.title("Green Channel")
        
        plt.subplot(2, 2, 4)
        plt.imshow(img_array[:,:,2], cmap='gray')
        plt.title("Blue Channel")
        
    else:
        # Grayscale image
        plt.imshow(img_array, cmap='gray')
        plt.title("Grayscale Mask")
        # Add colorbar to show values
        plt.colorbar(label="Pixel Value")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Diagnostic image saved to {output_path}")

def deep_debug_masks():
    """
    Perform deep debugging of mask files
    """
    script_dir = os.path.abspath(os.path.dirname(__file__))
    masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    images_dir = os.path.join(script_dir, 'train', 'train', 'images')
    
    # Create diagnostic folder
    debug_dir = os.path.join(script_dir, 'mask_debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Check if directories exist
    if not os.path.exists(masks_dir):
        print(f"ERROR: Masks directory {masks_dir} not found!")
        return
    
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory {images_dir} not found!")
    
    # Get all mask files
    mask_files = glob.glob(os.path.join(masks_dir, '*.*'))
    
    if not mask_files:
        print(f"No mask files found in {masks_dir}")
        return
    
    # Limit to 20 masks to avoid excessive output
    mask_files = mask_files[:20]
    print(f"Analyzing {len(mask_files)} mask files...")
    
    # Store mask info
    mask_info = []
    
    # Track different types
    rgb_masks = []
    grayscale_binary_masks = []
    grayscale_nonbinary_masks = []
    problem_masks = []
    
    for i, mask_file in enumerate(mask_files):
        try:
            mask_name = os.path.basename(mask_file)
            mask_path = os.path.abspath(mask_file)
            
            print(f"\n[{i+1}/{len(mask_files)}] Analyzing {mask_name}:")
            
            # Find corresponding image if it exists
            image_name = mask_name  # Assuming same filename
            image_path = os.path.join(images_dir, image_name)
            has_matching_image = os.path.exists(image_path)
            
            # Load the mask
            mask = Image.open(mask_file)
            mask_mode = mask.mode
            mask_format = mask.format
            mask_size = mask.size
            
            print(f"  Path: {mask_path}")
            print(f"  Format: {mask_format}, Mode: {mask_mode}, Size: {mask_size}")
            
            # Load as array for detailed analysis
            mask_array = np.array(mask)
            shape = mask_array.shape
            dimensions = len(shape)
            
            # Check dimensions
            if dimensions == 2:
                shape_info = f"2D grayscale: {shape[0]}x{shape[1]}"
                
                # Check unique values
                unique_values = np.unique(mask_array)
                print(f"  Shape: {shape_info}")
                print(f"  Unique values: {unique_values}")
                
                # Check if binary
                is_binary = len(unique_values) <= 2
                binary_type = "None"
                
                if is_binary:
                    if set(unique_values).issubset({0, 1}):
                        binary_type = "0/1 binary"
                        grayscale_binary_masks.append(mask_file)
                    elif set(unique_values).issubset({0, 255}):
                        binary_type = "0/255 binary"
                        grayscale_binary_masks.append(mask_file)
                    else:
                        binary_type = f"Non-standard binary with values {unique_values}"
                        problem_masks.append(mask_file)
                else:
                    grayscale_nonbinary_masks.append(mask_file)
                
                mask_type = "Grayscale " + binary_type if is_binary else f"Grayscale (non-binary) with {len(unique_values)} values"
                print(f"  Type: {mask_type}")
                
                # Create diagnostic image for non-standard masks
                if not is_binary or not set(unique_values).issubset({0, 1, 255}):
                    diag_path = os.path.join(debug_dir, f"{i+1}_{mask_name}_diagnostic.png")
                    save_diagnostic_image(mask_array, f"Grayscale Mask: {mask_name}", diag_path)
                
            elif dimensions == 3:
                channels = shape[2]
                shape_info = f"3D with {channels} channels: {shape[0]}x{shape[1]}x{shape[2]}"
                
                print(f"  Shape: {shape_info}")
                is_rgb = channels == 3
                is_rgba = channels == 4
                
                # Check if it's an RGB mask that's visually binary
                if is_rgb or is_rgba:
                    # Check if all channels are identical
                    if is_rgb:
                        channels_identical = np.array_equal(mask_array[:,:,0], mask_array[:,:,1]) and \
                                            np.array_equal(mask_array[:,:,0], mask_array[:,:,2])
                    else:  # RGBA
                        channels_identical = np.array_equal(mask_array[:,:,0], mask_array[:,:,1]) and \
                                            np.array_equal(mask_array[:,:,0], mask_array[:,:,2])
                    
                    if channels_identical:
                        mask_type = f"RGB mask with identical channels (visually grayscale, but stored as {'RGB' if is_rgb else 'RGBA'})"
                    else:
                        mask_type = f"RGB mask with different channels (true color mask, not binary)"
                    
                    rgb_masks.append(mask_file)
                else:
                    mask_type = f"Unusual 3D array with {channels} channels"
                    problem_masks.append(mask_file)
                
                print(f"  Type: {mask_type}")
                
                # Always save diagnostic for 3D masks
                diag_path = os.path.join(debug_dir, f"{i+1}_{mask_name}_diagnostic.png")
                save_diagnostic_image(mask_array, f"RGB Mask: {mask_name}", diag_path)
            
            else:
                shape_info = f"Unusual {dimensions}D shape: {shape}"
                print(f"  Shape: {shape_info}")
                mask_type = f"Unusual {dimensions}D array"
                problem_masks.append(mask_file)
            
            # Save a sample converted mask to show the difference 
            if dimensions == 3:
                # If it's RGB, convert to grayscale and save for comparison
                gray_mask = mask.convert('L')
                gray_array = np.array(gray_mask)
                unique_values = np.unique(gray_array)
                
                # Make it binary
                binary_array = (gray_array > 127).astype(np.uint8) * 255
                
                # Save the converted example
                converted_path = os.path.join(debug_dir, f"{i+1}_{mask_name}_converted.png")
                Image.fromarray(binary_array, mode='L').save(converted_path)
                print(f"  Converted grayscale sample saved to: {converted_path}")
            
            # Record complete info for this mask
            mask_info.append({
                'name': mask_name,
                'path': mask_path,
                'format': mask_format,
                'mode': mask_mode,
                'dimensions': dimensions,
                'shape': shape_info,
                'type': mask_type,
                'has_matching_image': has_matching_image,
                'image_path': image_path if has_matching_image else None
            })
            
        except Exception as e:
            print(f"  ERROR processing {mask_file}: {str(e)}")
            problem_masks.append(mask_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MASK ANALYSIS SUMMARY:")
    print("=" * 60)
    print(f"Total masks analyzed: {len(mask_files)}")
    print(f"RGB/RGBA masks: {len(rgb_masks)} (needs conversion to grayscale)")
    print(f"Grayscale binary masks: {len(grayscale_binary_masks)} (correct format)")
    print(f"Grayscale non-binary masks: {len(grayscale_nonbinary_masks)} (needs conversion to binary)")
    print(f"Problem masks: {len(problem_masks)}")
    
    # Generate detailed report
    report_path = os.path.join(debug_dir, "mask_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("MASK ANALYSIS DETAILED REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        for i, info in enumerate(mask_info):
            f.write(f"[{i+1}] {info['name']}\n")
            f.write(f"    Path: {info['path']}\n")
            f.write(f"    Format: {info['format']}, Mode: {info['mode']}\n")
            f.write(f"    Dimensions: {info['dimensions']}D, Shape: {info['shape']}\n")
            f.write(f"    Type: {info['type']}\n")
            if info['has_matching_image']:
                f.write(f"    Matching image: {info['image_path']}\n")
            else:
                f.write(f"    No matching image found\n")
            f.write("\n")
        
        # Add recommendations
        f.write("\nRECOMMENDATIONS:\n")
        if rgb_masks:
            f.write("1. Convert RGB masks to grayscale binary (black and white) format\n")
            f.write("   The following masks need conversion:\n")
            for mask in rgb_masks[:5]:  # Show first 5
                f.write(f"   - {os.path.basename(mask)}\n")
            if len(rgb_masks) > 5:
                f.write(f"   - ... and {len(rgb_masks) - 5} more\n")
        
        if grayscale_nonbinary_masks:
            f.write("\n2. Convert grayscale non-binary masks to binary (0/255) format\n")
            f.write("   The following masks need conversion:\n")
            for mask in grayscale_nonbinary_masks[:5]:
                f.write(f"   - {os.path.basename(mask)}\n")
            if len(grayscale_nonbinary_masks) > 5:
                f.write(f"   - ... and {len(grayscale_nonbinary_masks) - 5} more\n")
        
        if problem_masks:
            f.write("\n3. Fix or regenerate the following problematic masks:\n")
            for mask in problem_masks:
                f.write(f"   - {os.path.basename(mask)}\n")
    
    print(f"\nDetailed analysis report saved to: {report_path}")
    print(f"Diagnostic images saved to: {debug_dir}")
    
    # Return details for main function
    return {
        'rgb_masks': rgb_masks,
        'grayscale_binary_masks': grayscale_binary_masks,
        'grayscale_nonbinary_masks': grayscale_nonbinary_masks,
        'problem_masks': problem_masks,
        'debug_dir': debug_dir,
        'report_path': report_path
    }

def create_fix_script(analysis_result):
    """Create a specific fix script based on the mask analysis"""
    if not analysis_result:
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fix_script_path = os.path.join(script_dir, 'fix_these_masks.py')
    
    # Get mask lists
    rgb_masks = analysis_result['rgb_masks']
    grayscale_nonbinary_masks = analysis_result['grayscale_nonbinary_masks']
    problem_masks = analysis_result['problem_masks']
    
    # Only proceed if we have masks to fix
    if not (rgb_masks or grayscale_nonbinary_masks or problem_masks):
        print("\nNo masks need fixing!")
        return
    
    # Create a targeted fix script
    with open(fix_script_path, 'w') as f:
        f.write("""import os
import numpy as np
from PIL import Image
import glob

def fix_specific_masks():
    \"\"\"Fix specific masks identified as problematic\"\"\"
    # Create output directory for fixed masks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    fixed_dir = os.path.join(script_dir, 'train', 'train', 'masks_fixed')
    os.makedirs(fixed_dir, exist_ok=True)
    
    # Lists of masks to fix
""")
        
        # Add RGB masks
        if rgb_masks:
            f.write("    # RGB masks to convert to grayscale binary\n")
            f.write("    rgb_masks = [\n")
            for mask in rgb_masks:
                f.write(f"        r\"{os.path.basename(mask)}\",\n")
            f.write("    ]\n\n")
        else:
            f.write("    rgb_masks = []\n\n")
        
        # Add grayscale non-binary masks
        if grayscale_nonbinary_masks:
            f.write("    # Grayscale non-binary masks to convert to binary\n")
            f.write("    nonbinary_masks = [\n")
            for mask in grayscale_nonbinary_masks:
                f.write(f"        r\"{os.path.basename(mask)}\",\n")
            f.write("    ]\n\n")
        else:
            f.write("    nonbinary_masks = []\n\n")
        
        # Add problem masks
        if problem_masks:
            f.write("    # Problem masks to fix\n")
            f.write("    problem_masks = [\n")
            for mask in problem_masks:
                f.write(f"        r\"{os.path.basename(mask)}\",\n")
            f.write("    ]\n\n")
        else:
            f.write("    problem_masks = []\n\n")
        
        # Add the fix functions
        f.write("""    # Process RGB masks
    for mask_name in rgb_masks:
        try:
            mask_path = os.path.join(masks_dir, mask_name)
            output_path = os.path.join(fixed_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: {mask_path} not found")
                continue
            
            print(f"Converting RGB mask: {mask_name}")
            # Open the image
            img = Image.open(mask_path)
            # Convert to grayscale
            gray = img.convert('L')
            # Convert to binary (0/255)
            binary_array = np.array(gray)
            binary_array = (binary_array > 127).astype(np.uint8) * 255
            # Save
            Image.fromarray(binary_array, mode='L').save(output_path)
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"Error fixing {mask_name}: {e}")
    
    # Process grayscale non-binary masks
    for mask_name in nonbinary_masks:
        try:
            mask_path = os.path.join(masks_dir, mask_name)
            output_path = os.path.join(fixed_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: {mask_path} not found")
                continue
            
            print(f"Converting non-binary mask: {mask_name}")
            # Open the image
            img = Image.open(mask_path)
            # Convert to binary (0/255)
            binary_array = np.array(img)
            binary_array = (binary_array > 127).astype(np.uint8) * 255
            # Save
            Image.fromarray(binary_array, mode='L').save(output_path)
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"Error fixing {mask_name}: {e}")
    
    # Process problem masks
    for mask_name in problem_masks:
        try:
            mask_path = os.path.join(masks_dir, mask_name)
            output_path = os.path.join(fixed_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: {mask_path} not found")
                continue
            
            print(f"Fixing problem mask: {mask_name}")
            # Try to convert in the most general way
            img = Image.open(mask_path)
            # First convert to grayscale if it's not
            if img.mode != 'L':
                img = img.convert('L')
            # Then make it binary
            binary_array = np.array(img)
            binary_array = (binary_array > 127).astype(np.uint8) * 255
            # Save
            Image.fromarray(binary_array, mode='L').save(output_path)
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"Error fixing {mask_name}: {e}")
    
    # Count fixed masks
    fixed_count = len(glob.glob(os.path.join(fixed_dir, '*.*')))
    print(f"\nFixed {fixed_count} masks. All saved to {fixed_dir}")
    
    # Ask if user wants to replace originals
    print("\nDo you want to replace the original masks with the fixed ones? (y/n)")
    response = input("> ")
    
    if response.lower() == 'y':
        # Backup originals first
        backup_dir = os.path.join(script_dir, 'train', 'train', 'masks_backup')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy originals to backup
        for mask_path in glob.glob(os.path.join(masks_dir, '*.*')):
            mask_name = os.path.basename(mask_path)
            backup_path = os.path.join(backup_dir, mask_name)
            try:
                import shutil
                shutil.copy2(mask_path, backup_path)
            except Exception as e:
                print(f"Error backing up {mask_name}: {e}")
        
        print(f"Original masks backed up to {backup_dir}")
        
        # Replace with fixed masks
        for fixed_path in glob.glob(os.path.join(fixed_dir, '*.*')):
            fixed_name = os.path.basename(fixed_path)
            original_path = os.path.join(masks_dir, fixed_name)
            try:
                import shutil
                shutil.copy2(fixed_path, original_path)
            except Exception as e:
                print(f"Error replacing {fixed_name}: {e}")
        
        print("Original masks replaced with fixed ones")
    else:
        print("\nFixed masks are available at:")
        print(fixed_dir)
        print("You can manually copy them to replace the originals if needed")

if __name__ == "__main__":
    print("=== Fixing Specific Masks ===")
    fix_specific_masks()
    print("\nAfter fixing masks, try training with:")
    print("python train_simple.py --batch 1 --epochs 1")
""")
    
    print(f"\nCreated targeted fix script: {fix_script_path}")
    print("Run this script to fix only the specific problematic masks:")
    print(f"python {os.path.basename(fix_script_path)}")

def main():
    print("=== Advanced Mask Format Debugging ===")
    
    # Do deep analysis of masks
    analysis_result = deep_debug_masks()
    
    # Create specific fix script
    if analysis_result:
        create_fix_script(analysis_result)
    
    print("\n=== What To Do Next ===")
    print("1. Check the diagnostic images in the mask_debug folder")
    print("2. Read the detailed analysis report")
    print("3. Run the fix_these_masks.py script to fix problematic masks")
    print("4. After fixing, try training with minimal parameters:")
    print("   python train_simple.py --batch 1 --epochs 1")

if __name__ == "__main__":
    main() 