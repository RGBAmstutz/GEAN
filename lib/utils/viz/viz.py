import json
import numpy as np
from plot_3d_h36m_overlay import draw_to_batch
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load joint positions from the JSON file
with open('joint_positions.json', 'r') as file:
    joint_data = json.load(file)

# Extract input and output joint positions
motion_in_poses = np.array(joint_data['input_joint_positions'])
motion_out_poses = np.array(joint_data['output_joint_positions'])


def clip_outliers(joint_positions, threshold=2.0):
    mean = np.mean(joint_positions, axis=(0, 1))
    std = np.std(joint_positions, axis=(0, 1))
    clipped_positions = np.clip(joint_positions, mean - threshold * std, mean + threshold * std)
    return clipped_positions


motion_in_poses = clip_outliers(motion_in_poses)
motion_out_poses = clip_outliers(motion_out_poses)

# Call the function with the joint positions arrays
#inputs = draw_to_batch([motion_in_poses], alt_color=True, figsize=(10, 10))
#outputs = draw_to_batch([motion_out_poses], figsize=(10, 10))
gt_mo = draw_to_batch(smpl_joints_batch=motion_out_poses, gt=motion_in_poses, figsize=(10,10))


def save_frame_as_cropped_pdf(frame, name, crop_box=None):
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Plot the frame
    ax.imshow(frame)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure as an image to a BytesIO buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Load the image using PIL
    img = Image.open(buf)

    # Convert RGBA to RGB (if needed)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Apply the crop if a crop_box is provided
    if crop_box:
        img = img.crop(crop_box)
    else:
        # Automatically crop the image to the non-white content
        img = img.crop(img.getbbox())

    # Save the cropped image as a PDF
    pdf_filename = f'{name}.pdf'
    img.save(pdf_filename, "PDF")


def save_frames_as_individual_pdfs(frames, base_name, crop_box=None):
    for i in range(frames.shape[1]):  # Iterate over frames
        save_frame_as_cropped_pdf(frames[0, i], f'{base_name}_frame_{i + 1}', crop_box=crop_box)


def save_frames_as_gif(frames, name, crop_box=None, duration=500, loop=0):
    """
    Save frames as an animated GIF.

    Parameters:
        frames (numpy array): Array of images, expected shape [batch, num_frames, ...].
        name (str): Base name for the GIF file.
        crop_box (tuple, optional): (left, upper, right, lower) coordinates to crop the image.
        duration (int, optional): Duration between frames in milliseconds.
        loop (int, optional): Number of loops (0 means infinite).
    """
    images = []

    # Loop over each frame
    for i in range(frames.shape[1]):
        # Create a figure and axis to plot the frame
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(frames[0, i])
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

        # Save the figure to a BytesIO buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Load the image with PIL
        img = Image.open(buf)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Apply cropping: either a provided crop_box or auto-crop to non-white content
        if crop_box:
            img = img.crop(crop_box)
        else:
            img = img.crop(img.getbbox())

        images.append(img)

    # Save the list of images as an animated GIF
    gif_filename = f'{name}.gif'
    images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=duration, loop=loop)


# Define the crop_box as (left, upper, right, lower)
# crop_box = (700, 100, 1200, 800)  # Adjust these values based on your specific needs
crop_box = (820, 140, 1090, 790)
# Save frames as individual PDFs with the specified crop_box
#save_frames_as_individual_pdfs(inputs, 'inputs/input', crop_box=crop_box)
#save_frames_as_individual_pdfs(outputs, 'outputs/output', crop_box=crop_box)
save_frames_as_individual_pdfs(gt_mo, 'outputs/out', crop_box=crop_box)
# save_frames_as_gif(inputs, 'input', crop_box=crop_box, duration=200, loop=0)
# save_frames_as_gif(outputs, 'output', crop_box=crop_box, duration=200, loop=0)
