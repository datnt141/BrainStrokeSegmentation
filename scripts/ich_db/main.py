import os
from libs.logs import Logs
from libs.sysParams import SysParams
from libs.datasets.ich_db import ICH_DB
from libs.stackedCNNUnets import StackedCnnUNets
if __name__ == '__main__':
    num_folds  = 5
    fold_index = 1
    """=================================Clsnet Parameters============================================================"""
    ckpts      = os.path.join(os.getcwd(), 'ckpts', 'Fold_{}_of_{}'.format(fold_index, num_folds))
    cls_params = SysParams()
    seg_params = SysParams()
    cls_params.vckpts          = ckpts             # Checkpoint for storing data            (Fixed)
    cls_params.vmodel_name     = 'cCNN_tiny_tiny'  # Model name                             (*)
    cls_params.vinput_shape    = (64, 64, 1)       # Input image shape                      (*)
    cls_params.vnum_classes    = 2                 # Number of target classes               (Fixed)
    cls_params.vtime_steps     = 1                 # For time-sequence classification       (Fixed)
    cls_params.vlr             = 0.0001            # Initial Learning rate                  (Fixed)
    cls_params.vloss           = 'swCE'            # Name of loss method                    (Fixed)
    cls_params.vweights        = (0.45, 0.55)      # For using weighted cross entropy       (Fixed)
    cls_params.vnum_epochs     = 20                # Number of training epochs              (Fixed)
    cls_params.vbatch_size     = 128               # Size of batch                          (Fixed)
    cls_params.vdb_repeat      = 10                # Repeat dataset at single learing rate  (Fixed)
    cls_params.vcontinue       = False             # Continue training or not               (Fixed)
    cls_params.vflip_ud        = True              # Flip up-down in data augmentation      (Fixed)
    cls_params.vflip_lr        = True              # Flip left-right in data augmentation   (Fixed)
    cls_params.vcrop_ratio     = 0.0               # Crop ratio                             (Fixed)
    cls_params.vdebug          = False             # Debug flag                             (Fixed)
    """=================================Segnet Parameters============================================================"""
    seg_params.vckpts                    = ckpts         # Checkpoint for storing data              (Fixed)
    seg_params.vinput_shape              = (64, 64, 1)   # Input image shape                        (*)
    seg_params.vnum_classes              = 2             # Number of target classes                 (Fixed)
    seg_params.vseg_use_bn               = False         # Segnet parameter                         (*)
    seg_params.vseg_bblock_type          = 'residual'    # Segnet parameter                         (*)
    seg_params.vseg_short_cut_rule       = 'concat'      # Segnet parameter                         (*)
    seg_params.vseg_short_cut_manipulate = True          # Segnet parameter                         (*)
    seg_params.vlr                       = 0.0001        # Initial Learning rate                    (Fixed)
    seg_params.vloss                     = 'Dice'        # Name of loss method                      (Fixed) 'swCE'
    seg_params.vweights                  = (0.45, 0.55)  # For using weighted cross entropy         (Fixed)
    seg_params.vnum_epochs               = 20            # Number of training epochs                (Fixed)
    seg_params.vbatch_size               = 64            # Size of batch                            (Fixed)
    seg_params.vdb_repeat                = 20            # Repeat dataset at single learing rate    (Fixed)
    seg_params.vcare_background          = False         # Consider background as an object or not  (Fixed)
    seg_params.vflip_ud                  = True          # Flip up-down in data augmentation        (Fixed)
    seg_params.vflip_lr                  = True          # Flip left-right in data augmentation     (Fixed)
    seg_params.vcontinue                 = False         # Continue training or not                 (Fixed)
    seg_params.vdebug                    = False         # Debug flag                               (Fixed)
    """=============================================================================================================="""
    StackedCnnUNets.vckpts           = ckpts      # Checkpoint for storing data                                  (Fixed)
    StackedCnnUNets.vcls_params      = cls_params # Classification network parameters                            (Fixed)
    StackedCnnUNets.vseg_params      = seg_params # Segmentation network parameters                              (Fixed)
    StackedCnnUNets.vcls_isize       = (64, 64)   # Size of block for clsnet                                     (*)
    StackedCnnUNets.vseg_isize       = (64, 64)   # Size ò block for segnet                                      (*)
    StackedCnnUNets.vcls_strides     = (32, 32)   # Stride for taking blocks for clsnet                          (*)
    StackedCnnUNets.vseg_strides     = (32, 32)   # Stride for taking blocks for segnet                          (*)
    StackedCnnUNets.vcls_sgray_level = 50         # Threshold for removing dark blocks                           (Fixed)
    StackedCnnUNets.vcls_object_size = 500        # Threshold for deciding blocks with/without gnd object        (Fixed)
    StackedCnnUNets.vseg_object_size = 500        # Threshold for deciding blocks with/without gnd object        (Fixed)
    StackedCnnUNets.vcls_th          = 0          # Threshold for making decision (extension)                    (Fixed)
    StackedCnnUNets.vcls_lsm_factor  = 0          # Label smoothing factor. A number from 0 to 1                 (Fixed)
    StackedCnnUNets.vseg_lsm_factor  = 0          # Label smoothing factor. A number from 0 to 1                 (Fixed)
    StackedCnnUNets.vdebug           = False      # Debug flag                                                   (Fixed)
    trainer = StackedCnnUNets()                   #(Fixed)
    """Get sample dataset"""
    ich_dber = ICH_DB(i_tsize=(512,512),i_num_folds=num_folds)
    train_db, val_db = ich_dber.call(i_fold_index=fold_index)
    trainer.train(i_train_db=train_db, i_val_db=val_db)                     #(Fixed)
    """2D Evaluation"""
    trainer.eval(i_db=val_db)                                               #(Fixed)
    """3D Evaluation"""
    val_db = ich_dber.get_val_patient(i_fold_index=fold_index)
    trainer.eval3d(i_db=val_db)
    Logs.move_log(i_dst_path=ckpts)
"""=================================================================================================================="""