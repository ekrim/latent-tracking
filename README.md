#### Applications of latent space generative models to hand pose estimation

Requires PyTorch

To train the models and run the experiments:

```
git clone git@github.com:ekrim/latent-tracking.git
cd latent-tracking
python train_pose.py
python train_flow.py
python eval.py --display_pose --display_flow
```
