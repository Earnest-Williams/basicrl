#!/bin/bash

# --- Configuration ---
LOG_FILE="run_log.txt"
CORE_SCRIPT="core.py"
PROCESSOR_SCRIPT="processor.py"
SHAPER_SCRIPT="shaper.py"
CORE_OUT_JSON="generated_cave_contextual.json"
PROC_OUT_JSON="processed_cave_data.json"
SHAPER_OUT_ARROW="shaped_dungeon_map.arrow"

# --- Helper Function for Logging ---
log_exec() {
    COMMAND=$1
    STEP_NAME=$2
    echo "--- Running $STEP_NAME ($COMMAND) ---" | tee -a "$LOG_FILE"
    bash -c "$COMMAND" >> "$LOG_FILE" 2>&1
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "!!! ERROR: $STEP_NAME failed with status $STATUS. Check $LOG_FILE. !!!" | tee -a "$LOG_FILE"
        exit $STATUS
    fi
    echo "--- $STEP_NAME Completed ---" | tee -a "$LOG_FILE"
}

# --- Start Script ---
echo "Starting Pipeline Run..." | tee "$LOG_FILE"
date | tee -a "$LOG_FILE"
echo "Cleaning previous output files..." | tee -a "$LOG_FILE"
rm -f "$CORE_OUT_JSON" "$PROC_OUT_JSON" "$SHAPER_OUT_ARROW" debug_*.png stitched_debug_grid.png *.log

# --- Run Core Generator ---
log_exec "python $CORE_SCRIPT" "Core Generator"
echo -e "\n--- $CORE_SCRIPT Summary Snippets ---" >> "$LOG_FILE"
if [ -f "$CORE_OUT_JSON" ]; then
    echo "Generation Settings:" >> "$LOG_FILE"
    grep -A 10 '"generation_settings":' "$CORE_OUT_JSON" | head -n 11 >> "$LOG_FILE"
    echo -e "\nFeature Summary Line:" >> "$LOG_FILE"
    grep "Features generated:" "$LOG_FILE" | tail -n 1 >> "$LOG_FILE"
else
    echo "ERROR: $CORE_OUT_JSON not found after running $CORE_SCRIPT." >> "$LOG_FILE"
    exit 1
fi

# --- Run Processor ---
if [ ! -f "$CORE_OUT_JSON" ]; then
    echo "ERROR: Input $CORE_OUT_JSON not found for Processor." | tee -a "$LOG_FILE"
    exit 1
fi
log_exec "python $PROCESSOR_SCRIPT" "Processor"
echo -e "\n--- $PROCESSOR_SCRIPT Summary Snippets ---" >> "$LOG_FILE"
grep "Processor: Found" "$LOG_FILE" | tail -n 1 >> "$LOG_FILE"

# --- Run Shaper ---
if [ ! -f "$PROC_OUT_JSON" ]; then
    echo "ERROR: Input $PROC_OUT_JSON not found for Shaper." | tee -a "$LOG_FILE"
    exit 1
fi
log_exec "python $SHAPER_SCRIPT" "Shaper"
echo -e "\n--- $SHAPER_SCRIPT Summary Snippets ---" >> "$LOG_FILE"
grep "Grid Params:" "$LOG_FILE" | tail -n 1 >> "$LOG_FILE"
grep "Found .* non-rock cells." "$LOG_FILE" | tail -n 1 >> "$LOG_FILE"
grep "Found .* distinct chambers." "$LOG_FILE" | tail -n 1 >> "$LOG_FILE"
grep "Generated DataFrame with" "$LOG_FILE" | tail -n 1 >> "$LOG_FILE"

# --- Stitch Debug Images (Updated for Proper 2x2 Layout) ---
echo -e "\n--- Stitching Debug Images (2x2 Format) ---" | tee -a "$LOG_FILE"
IMG_INITIAL="debug_initial_grid.png"
IMG_TYPE="debug_type_grid.png"
IMG_DEPTH="debug_depth_grid.png"
IMG_FINAL="debug_final_grid.png"
IMG_STITCHED="stitched_debug_grid.png"
ROW1_TMP="row1_tmp.png"
ROW2_TMP="row2_tmp.png"

STITCH_CMD_AVAILABLE=false
if command -v magick > /dev/null 2>&1; then
    STITCH_CMD_AVAILABLE=true
    echo "Found ImageMagick (magick command)." >> "$LOG_FILE"
else
    echo "Warning: ImageMagick command 'magick' not found. Cannot stitch images." | tee -a "$LOG_FILE"
fi

if $STITCH_CMD_AVAILABLE && [ -f "$IMG_INITIAL" ] && [ -f "$IMG_TYPE" ] && [ -f "$IMG_DEPTH" ] && [ -f "$IMG_FINAL" ]; then
    echo "Running ImageMagick commands..." >> "$LOG_FILE"

    # Create Top Row (Initial + Type)
    magick \
      \( -background white -fill black -gravity center -size 512x30 caption:"$IMG_INITIAL" "$IMG_INITIAL" -append \) \
      \( -background white -fill black -gravity center -size 512x30 caption:"$IMG_TYPE"    "$IMG_TYPE"    -append \) \
      +append "$ROW1_TMP" >> "$LOG_FILE" 2>&1

    # Create Bottom Row (Depth + Final)
    magick \
      \( -background white -fill black -gravity center -size 512x30 caption:"$IMG_DEPTH" "$IMG_DEPTH" -append \) \
      \( -background white -fill black -gravity center -size 512x30 caption:"$IMG_FINAL"  "$IMG_FINAL"  -append \) \
      +append "$ROW2_TMP" >> "$LOG_FILE" 2>&1

    # Stitch top and bottom rows vertically
    magick "$ROW1_TMP" "$ROW2_TMP" -append "$IMG_STITCHED" >> "$LOG_FILE" 2>&1
    rm -f "$ROW1_TMP" "$ROW2_TMP"

    if [ $? -eq 0 ] && [ -f "$IMG_STITCHED" ]; then
        echo "Successfully created $IMG_STITCHED (2x2 format)" | tee -a "$LOG_FILE"
    else
        echo "Error during ImageMagick stitching. Check $LOG_FILE." | tee -a "$LOG_FILE"
    fi
else
    if $STITCH_CMD_AVAILABLE; then
        echo "Skipping stitching: One or more debug PNG files are missing." | tee -a "$LOG_FILE"
    fi
fi

# --- Finish ---
echo -e "\n--- run.sh Finished ---" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"
echo "Check $LOG_FILE for details and $IMG_STITCHED for visual output."

exit 0

