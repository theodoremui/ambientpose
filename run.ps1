#----------- ultralytics -----------
# python .\cli\detect.py --min-confidence 0.3 `
# --video .\data\video\video.avi `
# --output-dir .\outputs\video-ultralytics `
# --backend ultralytics

#----------- openpose -----------
python .\cli\detect.py --min-confidence 0.5 `
--video .\data\video\video.avi `
--output-dir .\outputs\video-openpose `
--backend openpose