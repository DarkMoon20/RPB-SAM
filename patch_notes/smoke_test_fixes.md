# SAMatch ACDC smoke test fixes

Date: 2026-06-29

Smoke test command:
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train_unimatch_acdc.py \
  --root_path ./data/ACDC \
  --max_iterations 500 \
  --batch_size 8

Fixes:
1. Fixed hard-coded config path in networks/net_factory.py:
   from /data/maia/gpxu/proj1/samatch/code/configs/swin_tiny_patch4_window7_224_lite.yaml
   to ./code/configs/swin_tiny_patch4_window7_224_lite.yaml

2. Fixed complex learning-rate bug in train_unimatch_acdc.py:
   replaced (1 - iter_num / max_iterations) ** 0.9
   with max(0.0, 1 - iter_num / max_iterations) ** 0.9

3. Installed missing dependencies:
   tensorboardX
   scikit-image
   SimpleITK / medpy if needed

Smoke test passed:
- training loop runs
- evaluation runs
- MeanDice printed
