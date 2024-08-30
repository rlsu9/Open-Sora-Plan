import os
import re
from moviepy.editor import VideoFileClip, clips_array, ColorClip
import argparse


def pad_clips(clips, num_columns, clip_size=(256, 256)):
    """Pad clips with blank clips to make the row size consistent."""
    num_clips = len(clips)
    padding_needed = num_columns - num_clips % num_columns if num_clips % num_columns != 0 else 0
    
    if padding_needed:
        blank_clip = ColorClip(size=clip_size, color=(0, 0, 0), duration=clips[0].duration)
        clips.extend([blank_clip] * padding_needed)
    
    return clips

def create_video_grid(mp4_files, num_columns, output='output.mp4'):
    clips = [VideoFileClip(f) for f in mp4_files]
    
    # Resize clips to the same height to fit nicely into the grid
    clip_size = (clips[0].size[0], 360)
    clips = [clip.resize(height=360) for clip in clips]
    
    # Pad clips to ensure the row is consistent
    padded_clips = pad_clips(clips, num_columns, clip_size)
    
    # Create a single row of video clips
    row = [padded_clips[i:i + num_columns] for i in range(0, len(padded_clips), num_columns)]
    
    # Write the result to a file
    grid = clips_array(row)
    grid.write_videofile(output, codec='libx264')

def main(args):
    picked_checkpoint_index = list(range(args.start, args.end, args.step_size))
    folder = args.folder
    video_subfoler_list = [
        f for f in os.listdir(folder) 
        if re.match(r'^checkpoint-(\d+)$', f) and int(f.split('-')[-1]) in picked_checkpoint_index]
    video_subfoler_list = sorted(video_subfoler_list, key=lambda x: int(x.split('-')[1]))
    original_vid_folder = "original_100"
    video_subfoler_list = [original_vid_folder] + video_subfoler_list

    print(video_subfoler_list)

    first_subfolder = video_subfoler_list[0]
    file_name_dict = {} 
    for file in os.listdir(os.path.join(folder, first_subfolder)):
        if file.endswith('.mp4'):
            file_name_dict[file] = []


    for video_subfoler in video_subfoler_list:
        for video in os.listdir(os.path.join(folder, video_subfoler)):
            if video.endswith('.mp4') and "Euler" not in video:
                file_name_dict[video].append(os.path.join(folder, video_subfoler, video))

    grid_save_folder = os.path.join(folder, 'iteration_step_grid')
    os.makedirs(grid_save_folder, exist_ok=True)
        
    for file_name, file_path_list in file_name_dict.items():
        num_columns = len(file_path_list)
        create_video_grid(file_path_list, num_columns, output=os.path.join(grid_save_folder, file_name + '_grid.mp4'))
    
    print(video_subfoler_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=None, help='Path to the folder that contains checkpoint-xxxx')
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--step_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
            
# ./Open-Sora-Plan/opensora/sample/prompt_inference_grid.py --folder "/nfs-stor/rlsu_e/sora_model_imporve/vae_model_checker/checkpoint_debug_result" --start=150000 --end=225001 --step_size=25000