"""
Week-12
----------------------------------------------------------------
- feature 검출
- 검출한 feature feature_extraction_dir에 이미지 저장
- feature matching 결과 feature_matching_dir에 저장
...
- Extrinsic(R,t) 추정 
- 카메라 센터 3-D 시각화
"""
import os


# ───────────────────────────────────────────────────────────────
# 0. 고정 파라미터 
# ───────────────────────────────────────────────────────────────
images_dir              = "/home/starry/workspace/works/threed_computer_vision/icpbl/captured_images"
feature_extraction_dir  = "/home/starry/workspace/works/threed_computer_vision/icpbl/feature_extraction"
feature_matching_dir  = "/home/starry/workspace/works/threed_computer_vision/icpbl/feature_matching"
os.makedirs(feature_extraction_dir, exist_ok=True)
os.makedirs(feature_matching_dir, exist_ok=True)

