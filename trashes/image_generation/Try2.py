import os
from PIL import Image, ImageDraw, ImageFont
from utils import *
from visualizations import *

# Read the data
df = read_file_to_dataframe('../../logs/20250723.txt')

# Group by 'method_name' and 'case_number'
grouped = df.groupby(['method_name', 'case_number'])

image_paths = []
titles = []

# Generate individual images and store paths
for (method, case), group in grouped:
    save_file = group['save_file'].iloc[0]
    img_path = f"./images/experiment_{method}_{case}.png"
    visualize_a_case(
        case_number=case,
        checkpoint_file_path=f"./checkpoints/{save_file}",
        visualization_file_path=img_path
    )
    image_paths.append((case, method, img_path))
    titles.append((method, case, img_path))

# Sort images by case
image_paths_sorted = sorted(image_paths, key=lambda x: x[0])

# Load images
images = [Image.open(p[2]) for p in image_paths_sorted]

# Prepare to draw titles
font = ImageFont.load_default()

# Create a new image to hold all subgraphs vertically with titles
total_height = sum(img.height + 30 for img in images)  # extra space for titles
max_width = max(img.width for img in images)

big_image = Image.new('RGB', (max_width, total_height), color='white')
draw = ImageDraw.Draw(big_image)

current_y = 0
for (case, method, path), img in zip(image_paths_sorted, images):
    # Draw title
    title_text = f"Method: {method}, Case: {case}"
    draw.text((10, current_y), title_text, font=font, fill='black')
    current_y += 20  # space for title
    # Paste image
    big_image.paste(img, (0, current_y))
    current_y += img.height

# Save the combined image
combined_image_path = '../../images/combined_comparison.png'
big_image.save(combined_image_path)

# Close images and delete individual files
for img, (_, _, path) in zip(images, image_paths_sorted):
    img.close()
    if os.path.exists(path):
        os.remove(path)

print(f"Combined image saved at {combined_image_path}")