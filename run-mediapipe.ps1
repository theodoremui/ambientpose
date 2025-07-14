$backend = 'mediapipe'
$model_pose = 'POSE_LANDMARKS'

python cli/detect.py `
    --video data/video/video.avi `
    --output-dir "outputs/video-$backend" `
    --overlay-video "outputs/video-$backend/overlay.mp4" `
    --backend $backend `
    --min-confidence 0.5 `
    --net-resolution 656x368 `
    --model-pose $model_pose `
    --toronto-gait-format `
    --extract-comprehensive-frames `
    --verbose