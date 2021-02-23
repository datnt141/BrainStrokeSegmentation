# This code loads the CT slices (grayscale images) of the brain-window for each subject in ct_scans folder then saves them to
# one folder (data\image).
# Their segmentation from the masks folder is saved to another folder (data\label).
import os
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from imageio import imwrite
from libs.datasets.nfolds import sNFolds
db_path = r'G:\roots\segmentations\brain_stroke\ICH\downloaded_db\ich_db'
"""=======================Author: datnt================================================================="""
class ICH_DB:
    debug = False
    def __init__(self,i_db_path=None,i_tsize=(512,512),i_num_folds=5):
        assert isinstance(i_num_folds,int)
        assert i_num_folds>1
        if i_db_path is None:
            self.db_path = db_path
        else:
            assert isinstance(i_db_path, str)  # Path to the ich_db dataset. EXAMPLE: './downloaded_db/ich_db'
            assert os.path.exists(i_db_path)  # Path to the ich_db dataset. EXAMPLE: './downloaded_db/ich_db'
            self.db_path      = i_db_path
        self.num_folds    = i_num_folds
        """Fixed parameters"""
        self.numSubj      = 82      #Fixed
        self.new_size     = i_tsize #Becareful with different size for resizing mask because it can make interpolation values
        self.window_specs = (40,120)#Fixed
        """Create directory for storing images for debuging"""
        if self.debug:
            self.train_path = os.path.join(os.getcwd(),'ich')
            self.image_path = os.path.join(self.train_path,'image')
            self.label_path = os.path.join(self.train_path,'label')
            if not os.path.exists(self.train_path):
                os.mkdir(self.train_path)
                os.mkdir(self.image_path)
                os.mkdir(self.label_path)
            else:
                pass
        else:
            self.train_path = None
            self.image_path = None
            self.label_path = None
        """NFold division. Images of same patient only in either training or validation set"""
        self.nfolds = sNFolds(i_num_folds=self.num_folds,i_num_samples=75)
    """Support function for resizing image"""
    @classmethod
    def imresize(cls,i_image,i_new_size):
        assert isinstance(i_image,np.ndarray)
        return np.array(Image.fromarray(i_image).resize(i_new_size))
    """Support function for custom ct_image normalization"""
    @classmethod
    def window_ct (cls, i_ct_scan, i_w_level=40, i_w_width=120):
        w_min = i_w_level - i_w_width / 2
        w_max = i_w_level + i_w_width / 2
        num_slices=i_ct_scan.shape[2]
        for s in range(num_slices):
            slice_s = i_ct_scan[:,:,s]
            slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #Customized Min-Max scaling Normalization
            slice_s[slice_s < 0]   = 0
            slice_s[slice_s > 255] = 255
            i_ct_scan[:,:,s] = slice_s.astype(np.uint8)
        print('ct_scan shape = ',i_ct_scan.shape)
        return i_ct_scan
    """Main function to extract data"""
    def get_data(self):
        """Read label stored in the '.csv' file"""
        counterI = 0
        data     = []
        hemorrhage_diagnosis_df = pd.read_csv(os.path.join(self.db_path,'hemorrhage_diagnosis_raw_ct.csv'))
        """Extract data"""
        for sNo in range(0+49, self.numSubj+49):
            if (sNo>58) and (sNo<66): #no raw data were available for these subjects
                continue
            else:
                #Loading the CT scan
                ct_dir_subj   = os.path.join(self.db_path,'ct_scans', "{0:0=3d}.nii".format(sNo))
                ct_scan_nifti = nib.load(str(ct_dir_subj))
                ct_scan       = np.asanyarray(ct_scan_nifti.dataobj)
                ct_scan       = self.window_ct(ct_scan, self.window_specs[0], self.window_specs[1])
                #Loading the masks
                masks_dir_subj = os.path.join(self.db_path,'masks', "{0:0=3d}.nii".format(sNo))
                masks_nifti    = nib.load(str(masks_dir_subj))
                masks          = np.asanyarray(masks_nifti.dataobj)
                idx            = hemorrhage_diagnosis_df.values[:,0]==sNo
                sliceNos       = hemorrhage_diagnosis_df.values[idx, 1]
                NoHemorrhage   = hemorrhage_diagnosis_df.values[idx, 7]
                assert ct_scan.shape[2]==masks.shape[2]
                assert ct_scan.shape[2]==sliceNos.size
                patient = {'name':'P_{}'.format(sNo),'ct_path':ct_dir_subj,'mask_path': masks_dir_subj,'images':ct_scan,'masks':masks}
                data.append(patient)
                print('{} --- Loaded subject index = {} with {} images (and masks)'.format(sNo,sNo,ct_scan.shape[2]))
                """Debuging images"""
                if self.debug:
                    for sliceI in range(0, sliceNos.size):
                        """Saving the a given CT slice"""
                        image = self.imresize(ct_scan[:,:,sliceI], self.new_size)
                        imwrite(os.path.join(self.image_path,'{}.png'.format(counterI)), image)
                        """Saving the segmentation for a given slice"""
                        mask = self.imresize(masks[:,:,sliceI], self.new_size)
                        imwrite(os.path.join(self.label_path,'{}.png'.format(counterI)), mask)
                        """Check for labeling of with/without hemorrhage"""
                        provided_label = NoHemorrhage[sliceI]
                        print('Provided label = ',provided_label)
                        sum_mask = np.sum(mask)
                        print('Sum Mask = {} => {}'.format(sum_mask,int(sum_mask==0)))
                        assert provided_label == int(sum_mask==0)
                        counterI = counterI+1
                else:
                    counterI += ct_scan.shape[2]
        print('Number of subjects = {} with {} images'.format(len(data),counterI))
        return data
    """Main function"""
    def call(self,i_fold_index=1):
        assert isinstance(i_fold_index, int)
        assert 0 < i_fold_index <= self.num_folds
        data = self.get_data()
        train_images, train_labels = [], []
        val_images, val_labels     = [], []
        train_data, val_data = self.nfolds(i_fold_index=i_fold_index, i_num_aug_samples=1)
        for index, item in enumerate(data):
            if index in train_data:
                item_images = item['images']
                item_masks = item['masks']
                for sliceInd in range(0, item_images.shape[2]):
                    image = item_images[:, :, sliceInd]  # original image
                    mask  = item_masks[:, :, sliceInd]   # original image
                    train_images.append(image)
                    train_labels.append(mask)
            elif index in val_data:
                item_images = item['images']
                item_masks  = item['masks']
                for sliceInd in range(0, item_images.shape[2]):
                    image = item_images[:, :, sliceInd]   # original image
                    mask  = item_masks[:, :, sliceInd]    # original mask
                    val_images.append(image)
                    val_labels.append(mask)
            else:
                pass  # raise Exception('Error?')
        train_images = np.array(train_images).astype(np.uint8)
        train_labels = np.array(train_labels).astype(np.uint8)
        train_labels = (train_labels>0).astype(np.uint8)
        val_images   = np.array(val_images).astype(np.uint8)
        val_labels   = np.array(val_labels).astype(np.uint8)
        val_labels   = (val_labels>0).astype(np.uint8)
        """Note: image is normal color images (0-255) and labels is index image (dytpe = np.uint8)"""
        return list(zip(train_images,train_labels)),list(zip(val_images,val_labels))
"""==========================================================================================="""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('This module is to extract images for ICH dataset')
    tester = ICH_DB(i_db_path=db_path,i_tsize=(512,512),i_num_folds=5)
    train_db, val_db = tester.call(i_fold_index=1)
    for train_item in train_db:
        train_image,train_mask = train_item
        if np.sum(train_mask)>0:
            plt.subplot(1,2,1)
            plt.imshow(train_image,cmap='gray')
            plt.title('Image')
            plt.subplot(1, 2, 2)
            plt.imshow(train_mask, cmap='gray')
            plt.title('Mask')
            plt.show()
        else:
            pass
        print('Shape = ', train_image.shape, train_mask.shape)
"""==========================================================================================="""