# Calibration

```bash
python tools/vision_arm_cal.py --collect # Use touchscreen to move, press 'q' to capture image (16 images total)
python tools/vision_arm_cal.py --calibrate # Estimate calibration and present reprojection results, and press enter.
python tools/vision_arm_cal.py --test # Online reprojection visualization
```
Results are stored in `data/calibration`.

# Vision Accuracy
Uncomment the settings for the lock or object of interest in `vision_accuracy.sh`. Then run:

```bash
bash tools/vision_accuracy.sh collect # Assume arm is aligned with hole, sample 200 random poses
# Estimate segmentation masks with SAM2
# left click for positive prompt, right click for negative prompt
# press q to advance and r to change prompts
bash tools/vision_accuracy.sh mask
bash tools/vision_accuracy.sh pose # Estimate poses with FoundationPose
bash tools/vision_accuracy.sh accuracy # Estimate statistics and produce a basic plot
```

# Vision Insertion
Uncomment the settings for the object of interest and set the trial number in `vision_insertion.sh`. Then run:

```bash
# Assume arm is aligned with hole, reproducibly sample a random pose
# Estimate segmentation masks with SAM2
# Move the key to the estimated lock pose
# Runs the extremum seeking insertion
bash tools/vision_insertion.sh collect
```
