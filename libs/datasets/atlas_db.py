import os
import gzip
import shutil
import numpy as np
import nibabel as nib
from libs.datasets.nfolds import sNFolds
from libs.commons import ListTuples,SupFns
"""=================================================================================================================="""
db_path = r'G:\roots\segmentations\brain_stroke\ATLAS\downloaded_db\pkg36684-0001_REST\ATLAS R 1.2\standard_MNI'
class ATLAS_STANDARD:
    def __init__(self,i_db_path=None,i_tsize = (256, 256), i_num_folds=5):
        assert isinstance(i_num_folds,int)
        assert i_num_folds>1
        assert isinstance(i_tsize,(int,list,tuple))
        if isinstance(i_tsize,int):
            assert i_tsize>0
            self.tsize = (i_tsize,i_tsize)
        else:
            assert len(i_tsize)==2
            assert i_tsize[0]>0
            assert i_tsize[1]>0
            self.tsize = i_tsize
        self.num_folds = i_num_folds
        if i_db_path is None:
            self.db_path = db_path
        else:
            assert isinstance(i_db_path,str), 'Got type: {}'.format(type(i_db_path))
            assert os.path.exists(i_db_path)
            self.db_path = i_db_path
        self.part_a = os.path.join(self.db_path,'standard_part1')
        self.part_b = os.path.join(self.db_path,'standard_part2')
        self.patients = self.list_patients()
        self.nfolds  = sNFolds(i_num_folds=self.num_folds,i_num_samples=len(self.patients))
    def list_patients(self):
        def get_subdirs(i_part_path=None):
            assert os.path.exists(i_part_path)
            cohorts = os.walk(i_part_path)
            for item in cohorts:
                dir_path, dir_names, _ = item
                if dir_path == i_part_path:
                    return dir_names
            raise Exception('Invalid path to cohort!')
        def get_patients(i_part_path=None):
            cohort_names = get_subdirs(i_part_path=i_part_path)
            all_patients = []
            for cohort_name in cohort_names:
                cohort_path = os.path.join(i_part_path,cohort_name)
                assert os.path.exists(cohort_path),'Got value: {}'.format(cohort_path)
                patients = get_subdirs(i_part_path=cohort_path)
                for patient in patients:
                    all_patients.append(os.path.join(cohort_path,patient))
            return all_patients
        part_a_patients = get_patients(i_part_path=self.part_a)
        part_b_patients = get_patients(i_part_path=self.part_b)
        db_patients = part_a_patients + part_b_patients
        print('Number of patients = {}'.format(len(db_patients)))
        return db_patients
    @classmethod
    def get_nii_data(cls, i_nii_file_path=None):
        assert isinstance(i_nii_file_path, str), 'Got type: {}'.format(type(i_nii_file_path))
        assert os.path.exists(i_nii_file_path), 'Got value: {}'.format(i_nii_file_path)
        """Extract data"""
        file_path, patient_id = os.path.split(i_nii_file_path)
        nii_files  = [file for file in os.listdir(i_nii_file_path) if file.endswith('.nii.gz')]
        image_file = None
        label_file = None
        for index, item in enumerate(nii_files):
            assert isinstance(item,str),'Got type: {}'.format(type(item))
            if item.find('_LesionSmooth_stx')>0:
                label_file = item
            else:
                image_file = item
        assert image_file is not None
        assert label_file is not None
        assert image_file.find(patient_id)>=0
        assert label_file.find(patient_id)>=0
        assert image_file.endswith('.nii.gz')
        assert label_file.endswith('.nii.gz')
        image_gz_file  = os.path.join(i_nii_file_path,image_file)
        image_nii_file = os.path.join(i_nii_file_path,image_file[0:len(image_file)-3])
        label_gz_file  = os.path.join(i_nii_file_path,label_file)
        label_nii_file = os.path.join(i_nii_file_path,label_file[0:len(label_file)-3])
        """Extract image file"""
        if not os.path.exists(image_nii_file):
            with gzip.open(image_gz_file, 'rb') as gzfile:
                with open(image_nii_file, 'wb') as exfile:
                    shutil.copyfileobj(gzfile, exfile)
        else:
            pass
        images = np.array(nib.load(image_nii_file).dataobj)
        """Extract label file"""
        if not os.path.exists(label_nii_file):
            with gzip.open(label_gz_file, 'rb') as gzfile:
                with open(label_nii_file, 'wb') as exfile:
                    shutil.copyfileobj(gzfile, exfile)
        else:
            pass
        labels = np.array(nib.load(label_nii_file).dataobj)
        assert np.sum(ListTuples.compare(images.shape,labels.shape))==0
        assert len(images.shape) == 3, 'Got value: {}'.format(images.shape)
        return images,labels
    def get_data(self,i_fold_index=1,i_axis=2):
        assert isinstance(i_fold_index,int), 'Got type: {}'.format(type(i_fold_index))
        assert 0<i_fold_index<=self.num_folds
        assert isinstance(i_axis,int) #Axis of 3D volume where we taking scan images
        assert i_axis in(0,1,2)       #As the 3D volume of scan images
        train_index,val_index = self.nfolds(i_fold_index=i_fold_index)
        train_images, train_labels = [],[]
        val_images, val_labels     = [],[]
        for index, patient in enumerate(self.patients):
            print('Loading index {}: {}'.format(index,patient))
            images, labels = self.get_nii_data(i_nii_file_path=patient)
            num_images = images.shape[i_axis]
            for img_index in range(num_images):
                if i_axis == 0:
                    image = images[img_index,:,:]
                    label = labels[img_index,:,:]
                elif i_axis == 1:
                    image = images[:,img_index,:]
                    label = labels[:, img_index, :]
                else:
                    image = images[:,:,img_index]
                    label = labels[:, :, img_index]
                image = SupFns.imresize(i_image=image,i_tsize=self.tsize)
                label = (label>0).astype(np.int)
                label = SupFns.scale_mask(i_mask=label,i_tsize=self.tsize)
                if index in train_index:
                    train_images.append(image)
                    train_labels.append(label)
                elif index in val_index:
                    val_images.append(image)
                    val_labels.append(label)
                else:
                    raise Exception()
        """Note: image is normal color images (0-255) and labels is index image (dytpe = np.uint8)"""
        return list(zip(train_images,train_labels)),list(zip(val_images,val_labels))
    def get_val_patient(self, i_fold_index=1,i_axis=2):
        assert isinstance(i_fold_index, int), 'Got type: {}'.format(type(i_fold_index))
        assert 0 < i_fold_index <= self.num_folds
        assert isinstance(i_axis, int)  # Axis of 3D volume where we taking scan images
        assert i_axis in (0, 1, 2)  # As the 3D volume of scan images
        train_index, val_index = self.nfolds(i_fold_index=i_fold_index)
        val_patients = []
        for index, patient in enumerate(self.patients):
            if index in val_index:
                print('Loading index {}: {}'.format(index, patient))
                val_images,val_labels = [],[]
                images, labels = self.get_nii_data(i_nii_file_path=patient)
                num_images = images.shape[i_axis]
                for img_index in range(num_images):
                    if i_axis == 0:
                        image = images[img_index, :, :]
                        label = labels[img_index, :, :]
                    elif i_axis == 1:
                        image = images[:, img_index, :]
                        label = labels[:, img_index, :]
                    else:
                        image = images[:, :, img_index]
                        label = labels[:, :, img_index]
                    image = SupFns.imresize(i_image=image, i_tsize=self.tsize)
                    label = (label > 0).astype(np.int)
                    label = SupFns.scale_mask(i_mask=label, i_tsize=self.tsize)
                    val_images.append(image)
                    val_labels.append(label)
                val_images = np.swapaxes(np.swapaxes(np.squeeze(np.array(val_images)),0,1),1,2)
                val_labels = np.swapaxes(np.swapaxes(np.squeeze(np.array(val_labels)),0,1),1,2)
                print(val_images.shape,val_labels.shape)
                val_patients.append((val_images,val_labels)) #Format: ((images,masks),...,(images,masks))
                cmp = ListTuples.compare(i_x=val_images.shape, i_y=val_labels.shape)
                assert np.sum(cmp) == 0
                assert np.min(val_labels)==0
                assert np.max(val_labels)==1
            else:
                pass
        """Note: image is normal color images (0-255) and labels is index image (dytpe = np.uint8)"""
        return val_patients
    def get_statistics(self):
        def cal_hist(i_image=None):
            assert isinstance(i_image, np.ndarray)
            assert len(i_image.shape) in (2, 3)
            if len(i_image.shape) == 2:
                image = np.expand_dims(i_image, axis=-1)
            else:
                image = i_image.copy()
            assert image.shape[-1] in (1, 3)
            assert image.dtype in (np.uint8,)
            hist, bins = np.histogram(a=image, bins=range(0, 256))
            return hist
        train_data,val_data = self.get_data(i_fold_index=1,i_axis=2)
        all_data = train_data + val_data
        histograms  = []
        for item in all_data:
            item_image,item_label = item
            fusion_image = item_image*item_label
            if np.sum(fusion_image)>0:
                histograms.append(cal_hist(i_image=fusion_image))
            else:
                pass
        histograms = np.array(histograms)
        histograms = np.sum(histograms,axis=0)
        histograms[0]  = 0 #Background
        histograms[-1] = 0 #White
        """Measure of mean and std of distribution"""
        sum_val  = 0
        sum_item = 0
        for index, item in enumerate(histograms):
            sum_val  += index*item
            sum_item += item
        mean_val = sum_val/sum_item
        sum_val  = 0
        for index, item in enumerate(histograms):
            sum_val += (index - mean_val)*(index - mean_val)*item
        std_val = np.sqrt(sum_val/sum_item)
        print(mean_val, std_val)
        plt.plot(histograms)
        plt.show()
        return histograms,mean_val,std_val
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module is to prepare raw data using standard ATLAS dataset')
    import matplotlib.pyplot as plt
    exampler = ATLAS_STANDARD()
    #exampler.get_statistics()
    train_db, val_db = exampler.get_data(i_fold_index=1,i_axis=2)
    for element in train_db:
        img, msk = element
        print(img.shape,msk.shape)
        if np.sum(msk)>2000:
            plt.subplot(1,3,1)
            plt.imshow(img,cmap='gray')
            plt.subplot(1,3,2)
            plt.imshow(msk,cmap='gray')
            plt.subplot(1,3,3)
            plt.imshow(img*msk,cmap='gray')
            plt.show()
        else:
            pass
"""=================================================================================================================="""