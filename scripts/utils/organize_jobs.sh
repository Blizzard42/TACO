#!/bin/bash

# Define the base paths
BASE_DIR="/nethome/gpatlin3/flash/TACO/third_party/Robotwin/eval_result/mar/9/mmd_search"
RUN_NAME="gs_2_ema_0.8_ens_0.9_0.1"
SRC_DIR="$BASE_DIR/$RUN_NAME"

echo "Starting to organize directories from: $SRC_DIR"

# Loop through each seed directory (0, 1, 2, 3, 4, 5)
for seed_dir in "$SRC_DIR"/*/; do
    # Skip if it's not a directory
    [ -d "$seed_dir" ] || continue 
    
    seed=$(basename "$seed_dir")
    echo "Processing Seed: $seed"

    # Loop through each task directory (adjust_bottle, etc.)
    for task_dir in "$seed_dir"/*/; do
        [ -d "$task_dir" ] || continue
        
        task=$(basename "$task_dir")
        
        # The static path inside each task
        TARGET_PATH="$task_dir/pi05/demo_clean/None"
        
        # Make sure the target path actually exists before proceeding
        if [ ! -d "$TARGET_PATH" ]; then
            echo "  Warning: $TARGET_PATH not found, skipping..."
            continue
        fi

        # Read the timestamp directories into an array, sorted chronologically
        # mapfile safely handles the spaces in the folder names
        mapfile -t timestamps < <(ls -1 "$TARGET_PATH" | sort)

        # Loop through the 5 timestamps
        for i in "${!timestamps[@]}"; do
            ts="${timestamps[$i]}"
            set_num=$((i + 1)) # This gives us 1 through 5
            
            # Define the new destination path for this specific set
            DEST_RUN_DIR="${SRC_DIR}_set_${set_num}"
            DEST_PATH="$DEST_RUN_DIR/$seed/$task/pi05/demo_clean/None"
            
            # Create the destination directory structure (-p creates parent dirs as needed)
            mkdir -p "$DEST_PATH"
            
            # Move the specific timestamp folder to its new set folder
            # Wrapping variables in quotes is crucial here because of the space in the timestamp!
            mv "$TARGET_PATH/$ts" "$DEST_PATH/"
            
            echo "  Moved '$ts' to set_$set_num"
        done
    done
done

echo "Done! Your jobs are now separated into 5 chronological sets."