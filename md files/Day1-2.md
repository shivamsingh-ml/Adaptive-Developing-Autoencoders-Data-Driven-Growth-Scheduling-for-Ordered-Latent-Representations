Initial audit status:

Repo located: davidvgr/developing-autoencoders, public GitHub repo with 184 commits and Python/Jupyter-heavy codebase.

Training loop
Main loop is in solver.py: train_conv, test_conv, train_vali_all_epochs_conv, and dev_train_vali_all_epochs_conv. It trains with MSE reconstruction loss and SGD.

Growth logic
Growth is schedule-driven, not adaptive. dev_train_vali_all_epochs_conv uses size_per_epoch(size_ls, n_epochs, type='step'), then rebuilds the ConvAutoencoder each epoch and transfers/extends bottleneck weights via develope_convAE.

Baseline scripts found
train_all_models.py trains AE, PCA-AE, Dev-AE, and Conv-Dev-AE. Standard AE uses latent dim 128; PCA-AE initializes the bottleneck with PCA and freezes it; Dev-AE uses fixed schedule [6]*6 + [10]*6 + [17]*7 + [29]*7 + [50]*8 + [85]*8 + [128]*18.

Missing vs your target setup
No evidence found for MI estimation, k-NN evaluation, or adaptive MI/validation-trigger hooks. A repo-scoped search for those terms returned no results. The existing “adaptive-ish” logic is convergence-based loss thresholding in dev_train_vali_converge_conv, using 0.99 before final size and 0.999 at final size.

Hardcoded hyperparameters found
batch_size=128, num_workers=6, num_train_runs=40, n_epochs=60, lr=0.1, CIFAR10 normalization (0.4914, 0.4822, 0.4465)/(0.2470, 0.2435, 0.2616), train/val split 80/20, default growth manner cell_division, latent max 128, PCA freeze 20 epochs, Conv-Dev growth multiplier 1.7, start sizes [4, 6, 10, 16, 24].