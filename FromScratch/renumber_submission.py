import os
import pandas as pd
import re

def renumber_submission(input_file, output_file):
    """
    Reorder submission rows so each image appears at the line number matching its ID.
    For example, image_9438 should appear at line 9439 (accounting for header).
    Also removes any .png extensions from image IDs.
    
    Args:
        input_file (str): Path to input submission CSV file
        output_file (str): Path to output CSV file with reordered rows
    """
    # Read the original CSV file
    print(f"Reading submission file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Number of rows
    n_rows = len(df)
    print(f"Found {n_rows} entries in the submission file")
    
    # Get the column names
    img_id_column = df.columns[0]  # Assume first column is image ID
    
    # Get the mask data - check which column contains the RLE encoding
    if 'rle_mask' in df.columns:
        mask_column = 'rle_mask'
    elif 'EncodedPixels' in df.columns:
        mask_column = 'EncodedPixels'
    else:
        # Assume the second column contains the mask data
        mask_column = df.columns[1]
    
    # Clean image IDs by removing .png extension
    cleaned_image_ids = []
    for img_id in df[img_id_column]:
        # Remove .png extension if present
        if str(img_id).lower().endswith('.png'):
            img_id = str(img_id)[:-4]  # Remove last 4 characters (.png)
        cleaned_image_ids.append(img_id)
    
    # Create a new dataframe with standard column names and cleaned IDs
    new_df = pd.DataFrame({
        'ImageId': cleaned_image_ids,
        'EncodedPixels': df[mask_column].values
    })
    
    # Extract image numbers using regex
    def extract_image_number(img_id):
        match = re.search(r'(\d+)', str(img_id))
        if match:
            return int(match.group(1))
        return -1  # Default if no number found
    
    new_df['image_number'] = new_df['ImageId'].apply(extract_image_number)
    
    # Create a sorted index dataframe with all possible image numbers
    if new_df['image_number'].min() >= 0:
        max_image_num = new_df['image_number'].max()
        
        # Create a template dataframe with all image numbers from 0 to max
        template_df = pd.DataFrame({
            'image_number': range(max_image_num + 1),
            'ImageId': [f"image_{i}" for i in range(max_image_num + 1)],
            'EncodedPixels': [''] * (max_image_num + 1)  # Empty mask as default
        })
        
        # Use the image_number as index for merging
        new_df.set_index('image_number', inplace=True)
        template_df.set_index('image_number', inplace=True)
        
        # Update template with actual data where it exists
        for idx in new_df.index:
            if idx in template_df.index:
                template_df.loc[idx, 'EncodedPixels'] = new_df.loc[idx, 'EncodedPixels']
                template_df.loc[idx, 'ImageId'] = new_df.loc[idx, 'ImageId']
        
        # Reset index for saving
        result_df = template_df.reset_index(drop=True)
    else:
        print("Warning: Could not extract valid image numbers. Keeping original order.")
        result_df = new_df.drop(columns=['image_number'])
    
    # Save the reordered dataframe to CSV
    print(f"Saving reordered submission to: {output_file}")
    result_df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorder submission.csv so image IDs match line numbers")
    parser.add_argument("--input", type=str, default="outputs/submission.csv",
                       help="Path to input submission CSV file")
    parser.add_argument("--output", type=str, default="outputs/submission_reordered.csv",
                       help="Path to output reordered CSV file")
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Reorder the submission file
    renumber_submission(args.input, args.output) 