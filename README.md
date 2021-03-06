#### Applications of latent space generative models to hand pose estimation

![Subject P1, sequence 5](plots/P1_5.gif)

#### Instructions

Requires PyTorch

```
git clone git@github.com:ekrim/latent-tracking.git
cd latent-tracking
mkdir MSRA
```

Download the [MSRA hand pose dataset](https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0) and extract it to the `MSRA/` directory.

MSRA dataset courtesy of:

Xiao Sun, Yichen Wei, Shuang Liang, Xiaoou Tang and Jian Sun. Cascaded Hand Pose Regression. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015

Run the experiments (pretrained models present in `models/`):
```
python train_pose.py
python train_flow.py
python eval.py --display_pose --display_flow
```
