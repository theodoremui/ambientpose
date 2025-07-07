#----------- ultralytics -----------
# python .\cli\detect.py --min-confidence 0.3 `
# --video .\data\video\video.avi `
# --output-dir .\outputs\video-ultralytics `
# --backend ultralytics

#----------- openpose -----------
# python .\cli\detect.py --min-confidence 0.5 `
# --video .\data\video\video.avi `
# --output-dir .\outputs\video-openpose `
# --backend openpose

python cli/detect.py `
    --video data/video/video.avi `
    --output-dir outputs/video-openpose `
    --overlay-video outputs/video-openpose/overlay.mp4 `
    --backend openpose `
    --confidence-threshold 0.5 `
    --net-resolution 656x368 `
    --model-pose COCO `
    --toronto-gait-format `
    --extract-comprehensive-frames `
    --verbose