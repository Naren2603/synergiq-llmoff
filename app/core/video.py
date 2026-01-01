"""Video generation module - creates video from summary text."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
import textwrap

from app.core.text_clean import to_display_text


def generate_video_with_opencv(
    text: str,
    output_path: str,
    audio_path: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_per_screen: float = 5.0,
    font_scale: float = 0.8,
) -> Optional[str]:
    """
    Generate a video from text using OpenCV and PIL.
    
    Args:
        text: Summary text to display in video
        output_path: Output video file path
        audio_path: Optional audio file to mux into video
        width: Video width
        height: Video height
        fps: Frames per second
        duration_per_screen: Seconds to show each screen of text
        font_scale: Font scale for text
    
    Returns:
        Path to generated video or None on failure
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Split text into chunks that fit on screen
        max_chars_per_line = 60
        max_lines_per_screen = 15
        
        # Wrap text
        wrapped_lines = []
        for paragraph in text.split('\n'):
            if paragraph.strip():
                wrapped_lines.extend(textwrap.wrap(paragraph, width=max_chars_per_line))
            else:
                wrapped_lines.append('')
        
        # Create screens
        screens = []
        current_screen = []
        for line in wrapped_lines:
            current_screen.append(line)
            if len(current_screen) >= max_lines_per_screen:
                screens.append(current_screen)
                current_screen = []
        
        if current_screen:
            screens.append(current_screen)
        
        if not screens:
            screens = [["No content to display"]]
        
        # Create video writer (without audio first)
        temp_output = output_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        # Generate frames for each screen
        for screen_lines in screens:
            # Create PIL image for text rendering
            pil_img = Image.new('RGB', (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(pil_img)
            
            # Try to use a better font, fall back to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
            except:
                font = ImageFont.load_default()
            
            # Draw title
            title = "Document Summary"
            draw.text((50, 50), title, fill=(0, 0, 0), font=font)
            
            # Draw text lines
            y_offset = 120
            line_height = 35
            for line in screen_lines:
                draw.text((50, y_offset), line, fill=(50, 50, 50), font=font)
                y_offset += line_height
            
            # Convert PIL image to OpenCV format
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Write frames for this screen (duration_per_screen seconds)
            num_frames = int(fps * duration_per_screen)
            for _ in range(num_frames):
                video_writer.write(frame)
        
        video_writer.release()
        
        # If audio is provided, mux it with the video
        if audio_path and os.path.exists(audio_path):
            try:
                from moviepy.editor import VideoFileClip, AudioFileClip
                
                video_clip = VideoFileClip(temp_output)
                audio_clip = AudioFileClip(audio_path)
                
                # Set audio to video
                final_clip = video_clip.set_audio(audio_clip)
                
                # Adjust video duration to match audio if needed
                if audio_clip.duration > video_clip.duration:
                    # Loop video to match audio duration
                    final_clip = final_clip.loop(duration=audio_clip.duration)
                    final_clip = final_clip.set_audio(audio_clip)
                
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    fps=fps,
                    logger=None
                )
                
                video_clip.close()
                audio_clip.close()
                final_clip.close()
                
                # Clean up temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
            except Exception as e:
                print(f"Failed to mux audio with video: {e}")
                # Use video without audio
                if os.path.exists(temp_output):
                    os.rename(temp_output, output_path)
        else:
            # No audio, just rename temp file
            if os.path.exists(temp_output):
                os.rename(temp_output, output_path)
        
        if os.path.exists(output_path):
            print(f"Video generated successfully: {output_path}")
            return output_path
        
        return None
        
    except Exception as e:
        print(f"Video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_video(
    text: str,
    output_path: str,
    audio_path: Optional[str] = None,
) -> Optional[str]:
    """
    Generate video from summary text.
    
    Args:
        text: Summary text to convert to video
        output_path: Path where video should be saved
        audio_path: Optional path to audio file to include in video
    
    Returns:
        Path to generated video file or None if generation failed
    """
    if not text or not text.strip():
        print("Empty text provided for video generation")
        return None

    text = to_display_text(text)
    
    return generate_video_with_opencv(text, output_path, audio_path)
