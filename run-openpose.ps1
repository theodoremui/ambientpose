$backend = 'openpose'
$model_pose = 'COCO'

python cli/detect.py `
    --video data/video/video.avi `
    --output-dir "outputs/video-$backend" `
    --overlay-video "outputs/video-$backend/overlay.mp4" `
    --backend $backend `
    --confidence-threshold 0.2 `
    --net-resolution 656x368 `
    --model-pose $model_pose `
    --toronto-gait-format `
    --extract-comprehensive-frames `
    --verbose