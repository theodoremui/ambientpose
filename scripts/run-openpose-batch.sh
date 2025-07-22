#!/bin/bash
# OpenPose Batch Processing Script (Bash version)
# Processes all videos in data/videos directory using OpenPose backend
# Author: Theodore Mui
# Date: 2025-01-25

# Default parameters
RESUME=false
DRY_RUN=false
OUTPUT_BASE_DIR="outputs/openpose-batch"
MIN_CONFIDENCE="0.5"
NET_RESOLUTION="656x368"
MODEL_POSE="BODY_25"
TORONTO_GAIT_FORMAT=true
EXTRACT_COMPREHENSIVE_FRAMES=true
VERBOSE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --min-confidence)
            MIN_CONFIDENCE="$2"
            shift 2
            ;;
        --net-resolution)
            NET_RESOLUTION="$2"
            shift 2
            ;;
        --model-pose)
            MODEL_POSE="$2"
            shift 2
            ;;
        --no-toronto-gait-format)
            TORONTO_GAIT_FORMAT=false
            shift
            ;;
        --no-extract-comprehensive-frames)
            EXTRACT_COMPREHENSIVE_FRAMES=false
            shift
            ;;
        --no-verbose)
            VERBOSE=false
            shift
            ;;
        --help)
            echo "OpenPose Batch Processing Script"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --resume                              Resume from previous run"
            echo "  --dry-run                             Show what would be processed without running"
            echo "  --output-dir DIR                      Output directory (default: outputs/openpose-batch)"
            echo "  --min-confidence CONF                 Minimum confidence threshold (default: 0.5)"
            echo "  --net-resolution RES                  Network resolution (default: 656x368)"
            echo "  --model-pose MODEL                    Pose model (default: BODY_25)"
            echo "  --no-toronto-gait-format              Disable Toronto gait format output"
            echo "  --no-extract-comprehensive-frames     Disable comprehensive frame extraction"
            echo "  --no-verbose                          Disable verbose output"
            echo "  --help                                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${GREEN}[OPENPOSE BATCH] Starting OpenPose batch processing...${NC}"
echo -e "${CYAN}[INFO] Processing all videos in data/videos directory${NC}"

# Validate environment
echo -e "${CYAN}[CHECK] Validating environment...${NC}"

# Check if venv exists, if not, create and activate it
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[INFO] Python venv not found. Creating with 'uv venv'...${NC}"
    uv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}[INFO] Activating Python venv...${NC}"
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to activate virtual environment${NC}"
        exit 1
    fi
fi

# Check if CLI exists
if [ ! -f "cli/detect.py" ]; then
    echo -e "${RED}[ERROR] cli/detect.py is missing. Please ensure the repository is fully cloned.${NC}"
    exit 1
fi

# Check if videos directory exists
if [ ! -d "data/videos" ]; then
    echo -e "${RED}[ERROR] data/videos directory is missing.${NC}"
    exit 1
fi

# Check OpenPose environment variable (support both OPENPOSE_HOME and OPENPOSEPATH)
OPENPOSE_HOME_VAR=${OPENPOSE_HOME:-$OPENPOSEPATH}

if [ -z "$OPENPOSE_HOME_VAR" ]; then
    echo -e "${YELLOW}[WARNING] Neither OPENPOSE_HOME nor OPENPOSEPATH environment variable is set.${NC}"
    echo -e "${YELLOW}[WARNING] OpenPose backend may fall back to other available backends.${NC}"
else
    echo -e "${GREEN}[INFO] OpenPose installation found: $OPENPOSE_HOME_VAR${NC}"
fi

# Get all video files
readarray -t VIDEO_FILES < <(find data/videos -name "*.mp4" | sort)
TOTAL_VIDEOS=${#VIDEO_FILES[@]}

if [ $TOTAL_VIDEOS -eq 0 ]; then
    echo -e "${RED}[ERROR] No MP4 files found in data/videos directory.${NC}"
    exit 1
fi

echo -e "${GREEN}[INFO] Found $TOTAL_VIDEOS video files to process${NC}"

# Create base output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Create progress tracking file
PROGRESS_FILE="$OUTPUT_BASE_DIR/processing_progress.txt"
COMPLETED_VIDEOS=()

if [ "$RESUME" = true ] && [ -f "$PROGRESS_FILE" ]; then
    readarray -t COMPLETED_VIDEOS < "$PROGRESS_FILE"
    echo -e "${YELLOW}[INFO] Resume mode enabled. Found ${#COMPLETED_VIDEOS[@]} previously completed videos.${NC}"
fi

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    echo -e "${MAGENTA}[DRY RUN] Would process the following videos:${NC}"
    for video in "${VIDEO_FILES[@]}"; do
        video_name=$(basename "$video")
        if [ "$RESUME" = true ] && [[ " ${COMPLETED_VIDEOS[*]} " =~ " ${video_name} " ]]; then
            echo -e "  ${YELLOW}[SKIP] $video_name (already completed)${NC}"
        else
            echo -e "  ${CYAN}[PROCESS] $video_name${NC}"
        fi
    done
    echo -e "${MAGENTA}[DRY RUN] Use without --dry-run to actually process videos.${NC}"
    exit 0
fi

# Process each video
PROCESSED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
START_TIME=$(date +%s)

echo -e "${GREEN}[START] Beginning batch processing at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}========================================${NC}"

for i in "${!VIDEO_FILES[@]}"; do
    video="${VIDEO_FILES[$i]}"
    video_name=$(basename "$video" .mp4)
    video_path="$video"
    current_index=$((i + 1))
    
    echo -e "${CYAN}[$current_index/$TOTAL_VIDEOS] Processing: $(basename "$video")${NC}"
    
    # Skip if already completed in resume mode
    if [ "$RESUME" = true ] && [[ " ${COMPLETED_VIDEOS[*]} " =~ " $(basename "$video") " ]]; then
        echo -e "  ${YELLOW}[SKIP] Already completed in previous run${NC}"
        ((SKIPPED_COUNT++))
        continue
    fi
    
    # Create output directory for this video
    video_output_dir="$OUTPUT_BASE_DIR/$video_name"
    
    # Build command arguments
    args=(
        "cli/detect.py"
        "--video" "$video_path"
        "--output-dir" "$video_output_dir"
        "--overlay-video" "$video_output_dir/overlay.mp4"
        "--backend" "openpose"
        "--min-confidence" "$MIN_CONFIDENCE"
        "--net-resolution" "$NET_RESOLUTION"
        "--model-pose" "$MODEL_POSE"
    )
    
    if [ "$TORONTO_GAIT_FORMAT" = true ]; then
        args+=("--toronto-gait-format")
    fi
    
    if [ "$EXTRACT_COMPREHENSIVE_FRAMES" = true ]; then
        args+=("--extract-comprehensive-frames")
    fi
    
    if [ "$VERBOSE" = true ]; then
        args+=("--verbose")
    fi
    
    echo -e "  ${CYAN}[RUN] Executing OpenPose detection...${NC}"
    
    # Execute the command
    if python "${args[@]}"; then
        echo -e "  ${GREEN}[SUCCESS] $(basename "$video") completed successfully${NC}"
        
        # Add to completed list
        echo "$(basename "$video")" >> "$PROGRESS_FILE"
        ((PROCESSED_COUNT++))
    else
        echo -e "  ${RED}[ERROR] $(basename "$video") failed${NC}"
        ((FAILED_COUNT++))
    fi
    
    # Show progress
    current_time=$(date +%s)
    elapsed=$((current_time - START_TIME))
    if [ $PROCESSED_COUNT -gt 0 ]; then
        avg_time_per_video=$((elapsed / PROCESSED_COUNT))
        remaining_videos=$((TOTAL_VIDEOS - current_index))
        estimated_time_remaining=$((avg_time_per_video * remaining_videos))
        
        echo -e "  ${YELLOW}[PROGRESS] Processed: $PROCESSED_COUNT | Skipped: $SKIPPED_COUNT | Failed: $FAILED_COUNT | Remaining: $remaining_videos${NC}"
        echo -e "  ${YELLOW}[TIME] Avg: $((avg_time_per_video / 60)) min/video | ETA: $(printf '%02d:%02d:%02d' $((estimated_time_remaining/3600)) $((estimated_time_remaining%3600/60)) $((estimated_time_remaining%60)))${NC}"
    else
        echo -e "  ${YELLOW}[PROGRESS] Processed: $PROCESSED_COUNT | Skipped: $SKIPPED_COUNT | Failed: $FAILED_COUNT${NC}"
    fi
    echo ""
done

# Final summary
end_time=$(date +%s)
total_time=$((end_time - START_TIME))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}[COMPLETE] Batch processing finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}[SUMMARY] Total time: $(printf '%02d:%02d:%02d' $((total_time/3600)) $((total_time%3600/60)) $((total_time%60)))${NC}"
echo -e "${GREEN}[SUMMARY] Videos processed: $PROCESSED_COUNT${NC}"
echo -e "${YELLOW}[SUMMARY] Videos skipped: $SKIPPED_COUNT${NC}"

if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "${RED}[SUMMARY] Videos failed: $FAILED_COUNT${NC}"
else
    echo -e "${GREEN}[SUMMARY] Videos failed: $FAILED_COUNT${NC}"
fi

if [ $((PROCESSED_COUNT + FAILED_COUNT)) -gt 0 ]; then
    success_rate=$(( (PROCESSED_COUNT * 100) / (PROCESSED_COUNT + FAILED_COUNT) ))
    echo -e "${GREEN}[SUMMARY] Success rate: $success_rate%${NC}"
fi

if [ $PROCESSED_COUNT -gt 0 ]; then
    avg_time=$((total_time / PROCESSED_COUNT / 60))
    echo -e "${GREEN}[SUMMARY] Average time per video: $avg_time minutes${NC}"
fi

echo -e "${CYAN}[OUTPUT] All results saved in: $OUTPUT_BASE_DIR${NC}"

if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "${YELLOW}[NOTE] Some videos failed to process. Check the logs for details.${NC}"
    echo -e "${YELLOW}[NOTE] You can use --resume to retry failed videos.${NC}"
fi

echo -e "${GREEN}[DONE] OpenPose batch processing complete!${NC}" 