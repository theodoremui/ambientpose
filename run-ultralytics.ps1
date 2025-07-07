$backend = 'ultralytics'
$model_pose = 'yolov8n-pose.pt'
$video_path = '../videos/OAW09-top.mp4'
# $video_path = 'data/video/video.avi'

python cli/detect.py `
    --video $video_path `
    --output-dir "outputs/video-$backend" `
    --overlay-video "outputs/video-$backend/overlay.mp4" `
    --backend $backend `
    --confidence-threshold 0.2 `
    --net-resolution 656x368 `
    --model-pose $model_pose `
    --toronto-gait-format `
    --extract-comprehensive-frames `
    --verbose