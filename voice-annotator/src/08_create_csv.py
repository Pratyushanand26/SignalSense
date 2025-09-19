import os
import pandas as pd
import argparse

# This script is rewritten to correctly process data for a single audio file,
# split it into five sessions, and append the results to a CSV file.

import pandas as pd

def analyze_text_cues(text):
    """
    Analyzes a block of text for behavioral cues using correct patterns.
    """
    # Count tokens based on the correct format found in the text file
    tremor_count = text.count('/tremor/')
    pause_count = text.count('/pause_')
    
    # Placeholder logic for fillers and repeated words
    filler_count = 0
    repeat_word_count = 0

    return {
        "pause": pause_count,
        "tremor": tremor_count,
        "filler": filler_count,
        "repeat_word": repeat_word_count
    }

def get_session_prosody(prosody_df, start_time, end_time):
    """
    Filters the prosody DataFrame to get only words within a session's time range
    and returns the average prosody features.
    
    This function is more robust to empty sessions.
    """
    # Filter words that fall within the session's time interval
    session_prosody = prosody_df[
        (prosody_df['start'] >= start_time) & (prosody_df['end'] <= end_time)
    ]

    # Handle cases where there are no words in a session by returning NaNs
    if session_prosody.empty:
        return {
            "pitch": pd.NA,
            "intensity": pd.NA,
            "jitter": pd.NA
        }
    
    # Safely calculate mean values, handling potential NaN
    pitch = session_prosody['f0_mean'].mean()
    intensity = session_prosody['rms_mean'].mean()
    jitter = session_prosody['jitter_approx'].mean()

    return {
        "pitch": pitch,
        "intensity": intensity,
        "jitter": jitter
    }
def main(audio_path, txt_path, prosody_path, output_csv):
    # Check for required files
    if not os.path.exists(txt_path) or not os.path.exists(prosody_path):
        print(f"Error: Required files not found for {audio_path}. Skipping.")
        return

    # Extract shadow_id from the file name
    base_name = os.path.basename(audio_path)
    shadow_id = os.path.splitext(base_name)[0]

    # Read the ML-ready text file and split into sessions
    with open(txt_path, 'r') as f:
        # Assuming each line in the ml_ready.txt corresponds to a session
        session_texts = f.read().strip().split('\n')

    # Read the per-word prosody data
    prosody_df = pd.read_csv(prosody_path)

    # Calculate session start and end times to segment the prosody data
    # We will use the timestamps from the text file's first word for each line.
    session_times = []
    for text_line in session_texts:
        parts = text_line.strip().split(' ', 2)
        if len(parts) >= 2 and parts[0].startswith('[') and parts[0].endswith(']'):
            try:
                start_time = float(parts[0][1:-1])
                session_times.append(start_time)
            except (ValueError, IndexError):
                # Fallback if timestamp is not in the expected format
                session_times.append(0.0)
        else:
            session_times.append(0.0)

    session_start_end_times = []
    for i, start_time in enumerate(session_times):
        # The end time of a session is the start time of the next session,
        # or the end of the last word for the final session.
        if i + 1 < len(session_times):
            end_time = session_times[i+1]
        else:
            end_time = prosody_df['end'].max()
        
        session_start_end_times.append((start_time, end_time))

    # Compile data for each session
    data_to_append = []
    for i, session_text in enumerate(session_texts):
        # Get start and end times for the current session
        start_time, end_time = session_start_end_times[i]

        # Analyze text for cues
        text_features = analyze_text_cues(session_text)
        
        # Aggregate prosody data for the session
        prosody_features = get_session_prosody(prosody_df, start_time, end_time)

        # Create a single row of data for the session
        row = {
            "audio_file": os.path.basename(audio_path),
            "shadow_id": shadow_id,
            "session_id": i + 1,
            "text": session_text,
            "pause": text_features['pause'],
            "tremor": text_features['tremor'],
            "filler": text_features['filler'],
            "repeat_word": text_features['repeat_word'],
            "pitch": prosody_features['pitch'],
            "intensity": prosody_features['intensity'],
            "jitter": prosody_features['jitter']
        }
        data_to_append.append(row)

    # Create a DataFrame for all 5 sessions of the current audio file
    new_df = pd.DataFrame(data_to_append)

    # Append to CSV
    if os.path.exists(output_csv):
        new_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        new_df.to_csv(output_csv, mode='w', header=True, index=False)
    
    print(f"Successfully appended data for {os.path.basename(audio_path)} to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Appends audio analysis data to a CSV.")
    parser.add_argument("--audio_path", required=True, help="Path to the original audio file.")
    parser.add_argument("--txt_path", required=True, help="Path to the ML-ready text file.")
    parser.add_argument("--prosody_path", required=True, help="Path to the prosody features CSV.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file.")
    
    args = parser.parse_args()
    main(args.audio_path, args.txt_path, args.prosody_path, args.output_csv)