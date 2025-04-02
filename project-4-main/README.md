# Project 4: Semantic Segmentation Deep Learning

## Getting Started
  - See Project 0 for detailed environment setup.
  - Ensure that you are using the environment `cv_proj4`, which you can install using the install script `conda/install.sh`. 
 (Recommended)
- Alternative way to set up the environment (Recommended):
  1. make sure you are in your Project4/conda folder, then enter those commands in the terminal.
  2. ```mamba env create -f environment.yml```
  3. ```conda activate cv_proj4```
  4. go back to your Project4 folder, ```pip install -e .```
- If both methods above don't work well, you can try to create the environment by using conda and pip install:
  1. make sure you are in the Project4/conda folder, then enter those commands in the terminal.
  2. ```conda create -n cv_proj4 python=3.11.0```
  3. ```conda activate cv_proj4```
  4. ```pip install -r requirements.txt```
  5. go back to your Project4 folder, ```pip install -e .```

---

## PACE Support
- We've added [PACE ICE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102) support for this course which gives you access to GPUs. Refer to docs/PACE_guide.pdf for setup instructions.

## Logistics
- Submit via [Gradescope](https://gradescope.com).
- Part 6 (transfer learning) of this project is **optional** (extra credit) for 4476 and **required** for 6476.
- Additional information can be found in `docs/project-4.pdf`.

## Important Notes
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).


## 6476 Rubric

- +70 pts: Code
  
Note: +25 pts is dedicated towards your PSPNet model accuracy. You'll need to reach 60\% mIoU on your submitted grayscale PNG 
label maps to get full credit. You will get partial credit if your mIoU is lower than the threshold. score = mIoU/threshold x 25

- +30 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `grayscale_predictions.zip`: after running Colab, this zip file will be created with your model's outputs on the Camvid validation set (remember to download before closing your session!)
    - `exp/kitti/PSPNet/model/kitti_result/` - (optional for 4476) directory containing KITTI transfer learning results (remember to download before closing your session!)
    - `additional_data/`: (optional) if you use any data other than the images we provide you, please include them here
    - `README.txt`: (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g., any extra credit implementations), please describe what you did and how we can run the code. We will not award any extra credit if we can't run your code and verify the results.
2. `<your_gt_username>_proj4.pdf` - your report

## FAQ

### I'm getting [*insert error*] and I don't know how to fix it.

Please check [StackOverflow](https://stackoverflow.com/) and [Google](https://google.com/) first before asking the teaching staff about OS specific installation issues (because that's likely what we will be doing if asked).
