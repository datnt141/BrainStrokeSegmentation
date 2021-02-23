import os
import numpy as np
from libs.logs import Logs
import matplotlib.pyplot as plt
from libs.commons import SupFns
from libs.sysParams import SysParams
from libs.clsnets import ImageClsNets
from libs.segnets import ImageSegNets
from libs.metrics import SegMetrics_2D
from libs.datasets.tfrecords import TFRecordDB
np_int_types = (np.int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
"""=================================================================================================================="""
class StackedCnnUNets:
    vckpts           = None           # Checkpoint directory
    vcls_params      = None           # Clsnet parameter
    vseg_params      = None           # Segnet parameters
    vcls_isize       = (32, 32)       # Size of block for clsnet
    vseg_isize       = (64, 64)       # Size ò block for segnet
    vcls_strides     = (16, 16)       # Stride for taking blocks for clsnet
    vseg_strides     = (16, 16)       # Stride for taking blocks for segnet
    vcls_sgray_level = 0              # Threshold for removing dark blocks (Smallest gray level of considered blocks)
    vcls_object_size = 100            # Threshold for deciding blocks with/without ground-truth object
    vseg_object_size = 100            # Threshold for deciding blocks with/without ground-truth object
    vcls_th          = 0              # Threshold for making decision (extension)
    vcls_lsm_factor  = 0              # Label smoothing factor. A number from 0 to 1
    vseg_lsm_factor  = 0              # Label smoothing factor. A number from 0 to 1
    vdebug           = False          # Debug flag
    def __init__(self):
        self.clsnet = ImageClsNets()
        self.clsnet.init_params(i_params=self.vcls_params)
        self.segnet = ImageSegNets()
        self.segnet.init_params(i_params=self.vseg_params)
        """Inference parameters"""
        if self.vckpts is None:
            self.vckpts = os.path.join(os.getcwd(),'ckpts')
        else:
            pass
        self.clsnet.vckpts = self.vckpts
        self.segnet.vckpts = self.vckpts
        """Assertion"""
        assert isinstance(self.vcls_strides,(list,tuple,int))
        if isinstance(self.vcls_strides,(list,tuple)):
            assert len(self.vcls_strides)==2
            assert self.vcls_strides[0]>0
            assert self.vcls_strides[1]>0
        assert isinstance(self.vseg_strides,(list,tuple,int))
        if isinstance(self.vseg_strides,(list,tuple)):
            assert len(self.vseg_strides)==2
            assert self.vseg_strides[0] > 0
            assert self.vseg_strides[1] > 0
        assert isinstance(self.vseg_object_size,int)
        assert self.vseg_object_size>0
        assert isinstance(self.vcls_sgray_level,int)
        assert self.vcls_sgray_level>=0
    def init_params(self,i_params):
        assert isinstance(i_params, (SysParams,dict))
        Logs.log('-'*100)
        Logs.log('Init stacked cls-seg class global variables...')
        """As my design, all global variables of a class start with 'v' letter """
        global_variables = [var for var in self.__dict__ if var.startswith('v')]  # As my design
        if isinstance(i_params,SysParams):
            params = i_params.__dict__
        else:
            params = i_params.copy()
        for key in params.keys():
            val = params[key]
            if key in global_variables:
                Logs.log('Variable {} was changed from {} to : {}'.format(key,self.__dict__[key],val))
                self.__dict__.update({key: val})
            else:
                assert isinstance(key, str)
                if key.startswith('v'):
                    self.__dict__.update({key: val})
                    Logs.log('Init {} as : {}'.format(key, val))
                else:
                    pass
        """Custom parameter adjustment"""
        Logs.log('Set-up parameters for segmentation nets:')
        Logs.log_cls_params(self)
        self.__init__()
        return True
    def train(self,i_train_db=None,i_val_db=None):
        """Prepare data for clsnet according to clsnet params"""
        assert isinstance(i_train_db,(list, tuple)) # List of (image, mask, label) => Used for both cls and segnet cases
        cls_train_db = self.load_data(i_db=i_train_db,i_cls_flag=True,i_train_flag=True)
        cls_val_db   = self.load_data(i_db=i_val_db,i_cls_flag=True,i_train_flag=False)
        self.clsnet.train(i_train_db=cls_train_db,i_val_db=cls_val_db)
        """Prepaere data for segnet according to segnet params"""
        seg_train_db = self.load_data(i_db=i_train_db,i_cls_flag=False,i_train_flag=True)
        seg_val_db   = self.load_data(i_db=i_val_db,i_cls_flag=False,i_train_flag=False)
        """Adjusting new input size"""
        self.segnet.train(i_train_db=seg_train_db,i_val_db=seg_val_db)
    def eval(self,i_db=None):
        assert isinstance(i_db,(list,tuple)),'Got type: {}'.format(type(i_db))
        labels, preds = [],[]
        for index, element in enumerate(i_db):
            print('Evaluating index = {}'.format(index))
            image, mask = element  # As my design
            pred_image = self.predict(i_image=image)
            preds.append(pred_image)
            labels.append(mask)
        """Performance measurement"""
        evaluer = SegMetrics_2D(i_num_classes=self.segnet.vnum_classes, i_care_background=self.segnet.vcare_background)
        Logs.log('Using entire dataset')
        measures, measure_mean, measure_std = evaluer.eval(i_labels=labels, i_preds=preds, i_object_care=False)
        Logs.log('Measure shape = {}'.format(measures.shape))
        Logs.log('Measure mean  = {}'.format(measure_mean))
        Logs.log('Measure std   = {}'.format(measure_std))
        Logs.log('Using sub dataset that only consider images containing objects')
        measures, measure_mean, measure_std = evaluer.eval(i_labels=labels, i_preds=preds, i_object_care=True)
        Logs.log('Measure shape = {}'.format(measures.shape))
        Logs.log('Measure mean  = {}'.format(measure_mean))
        Logs.log('Measure std   = {}'.format(measure_std))
        return labels,preds
    """Predictions"""
    def cls_predict(self,i_image=None):
        """Note that: clsnet here is used to classify an image block into two classess of with or without objects"""
        """Classification of a single RGB image"""
        assert isinstance(i_image, np.ndarray), 'Got type: {}'.format(type(i_image))
        """Support function for making prediction block image"""
        def make_pred_image(i_pred_label=0.):
            assert isinstance(i_pred_label,(float,int)), 'Got type: {}'.format(type(i_pred_label))
            if i_pred_label>0:
                return np.ones(shape=(self.vcls_isize[0],self.vcls_isize[1],1),dtype=np.int)
            else:
                return np.zeros(shape=(self.vcls_isize[0],self.vcls_isize[1],1),dtype=np.int)
        assert len(i_image.shape) in (2,3), 'Got shape: {}'.format(i_image.shape)
        if len(i_image.shape) == 2: #Gray image with shape (height, width)
            i_image = np.expand_dims(i_image,-1)
        else:#RGB image with shape (height, width, depth)
            assert len(i_image.shape)==3
        assert i_image.shape[-1] in (1,3), 'Got shape: {}'.format(i_image.shape)   #Only gray or color images are accepted
        assert i_image.dtype in (np.uint8,), 'Got dtype: {}'.format(i_image.dtype) #Only accept normal RGB image (0~255)
        """Block extraction"""
        height, width, depth  = i_image.shape
        mask = np.zeros(shape =(height, width, 1))
        blocks, masks  = self.get_blks(i_image=i_image, i_mask=mask, i_blk_sizes=self.vcls_isize,i_blk_strides=self.vcls_strides)
        num_blk_height = len(blocks)
        num_blk_width  = len(blocks[0])
        blocks = self.forward_block_convert(blocks)
        blocks = np.array(blocks)         # Shape: (None, blk_height, blk_width, nchannels)
        assert len(blocks.shape)== 4 ,'Got shape: {}'.format(blocks.shape)      # Shape: (None, blk_height, blk_width, nchannels)
        assert blocks.shape[-1] == depth, 'Got shape: {}'.format(blocks.shape)  # Shape: (None, blk_height, blk_width, nchannels)
        """Preclassify based on block gray level"""
        block_means     = [np.mean(blk) for blk in blocks]
        pre_pred_labels = [x > self.vcls_sgray_level for x in block_means]
        pre_pred_labels = np.array(pre_pred_labels, dtype=np.float)
        """Prediction. DONOT Normalize data"""
        preds  = self.clsnet.predict(i_image=blocks) #Shape: (None, num_classes) where None is number of blocks extracted from image
        """As my design, preds has shape of (None, 2) as indicates the with/without existance of object in block"""
        pred_labels = (preds[:, 1] - preds[:, 0]) > self.vcls_th #self.vcls_th = 0 for conventional case (i. e. argmax)
        assert isinstance(pred_labels,np.ndarray)
        assert len(pred_labels.shape) == 1 #Shape (None, )
        pred_labels = pred_labels.astype(np.float)
        pred_labels = pred_labels * pre_pred_labels
        pred_blocks = [make_pred_image(i_pred_label=i) for i in pred_labels]
        pred_blocks = self.backward_block_convert(i_blocks=pred_blocks,i_height=num_blk_height,i_width=num_blk_width)
        pred_image  = self.join_blks(i_blks=pred_blocks, i_steps=self.vcls_strides) #Binary mask image with value of 0s and 1s
        print('cls pred_image shape = ',pred_image.shape)
        return preds, pred_image
    def seg_predict(self,i_image=None):
        """Note that: The segmentation problem has no overlapped objects (what can happened in detection problem)"""
        assert isinstance(i_image, np.ndarray), 'Got type: {}'.format(type(i_image))
        assert len(i_image.shape) in (2, 3), 'Got shape: {}'.format(i_image.shape)
        if len(i_image.shape) == 2:  # Gray image with shape (height, width)
            i_image = np.expand_dims(i_image, -1)
        else:  # RGB image with shape (height, width, depth)
            assert len(i_image.shape) == 3
        assert i_image.shape[-1] in (1, 3)  # Only gray or color images are accepted
        assert i_image.dtype in (np.uint8,)  # Only accept normal RGB image (0~255)
        """Block extraction"""
        height, width, depth= i_image.shape
        mask = np.zeros(shape=(height, width, 1))
        blocks, masks = self.get_blks(i_image=i_image, i_mask=mask, i_blk_sizes=self.vseg_isize,i_blk_strides=self.vseg_strides)
        num_blk_height = len(blocks)
        num_blk_width  = len(blocks[0])
        blocks = self.forward_block_convert(blocks)
        blocks = np.array(blocks)
        assert len(blocks.shape) == 4, '{}'.format(blocks.shape)  # Shape: (None, blk_height, blk_width,nchannels)
        """Prediction. DONOT Normalize data"""
        preds = self.segnet.predict(i_image=blocks)               # N-by256-by-256-by-1 for example.
        """Scaling preds to be same as the original"""
        spreds = []
        for pred in preds:
            spreds.append(SupFns.scale_mask(i_mask=pred,i_tsize=self.vseg_isize))
        preds      = self.backward_block_convert(spreds,num_blk_height,num_blk_width)
        pred_image = self.join_blks(i_blks=preds, i_steps=self.vseg_strides,i_overlapped_adjust=True)
        return pred_image #Shape (height, width,1) with gray level from 0 to (self.segnet.vnum_classes -1).
    def predict(self,i_image=None):
        assert isinstance(i_image,np.ndarray), 'Got type: {}'.format(type(i_image))
        """Prediction"""
        cls_pred = self.cls_predict(i_image=i_image)[-1]
        seg_pred = self.seg_predict(i_image=i_image)
        """Combining predictions"""
        pred_image = (cls_pred * seg_pred).astype(np.uint8)
        if self.vdebug:
            plt.subplot(1, 4, 1)
            plt.imshow(i_image)
            plt.subplot(1, 4, 2)
            plt.imshow(cls_pred)
            plt.subplot(1, 4, 3)
            plt.imshow(seg_pred)
            plt.subplot(1, 4, 4)
            plt.imshow(pred_image)
            plt.show()
        else:
            pass
        return pred_image
    """Support functions"""
    @classmethod
    def init_ckpts(cls, i_num_folds=5, i_fold_index=1):
        assert isinstance(i_num_folds, int)
        assert isinstance(i_fold_index, int)
        assert 0 < i_fold_index <= i_num_folds
        ckpts = os.path.join(os.getcwd(), 'ckpts', 'Fold_{}_of_{}'.format(i_fold_index, i_num_folds))
        if os.path.exists(ckpts):
            pass
        else:
            os.makedirs(ckpts)
        return ckpts
    @classmethod
    def get_blks(cls, i_image=None, i_mask=None, i_blk_sizes=None, i_blk_strides=None):
        """Note: Outputs are list of list that indicates the number of blocks in row and cols,respectively"""
        """Input image must be gray (h,w,1) or color (h,w,3)"""
        """Mask image must be gray (h,w)"""
        assert isinstance(i_image, np.ndarray), '{}'.format(type(i_image))
        assert isinstance(i_blk_sizes, (list, tuple, int)), '{}'.format(type(i_blk_sizes))
        assert isinstance(i_blk_strides, (list, tuple, int)),'{}'.format(type(i_blk_strides))
        assert len(i_image.shape) in (2, 3), '{}'.format(i_image.shape)
        if len(i_image.shape) == 2:  # For the case of i_image is gray with shape (height, width)
            image = np.expand_dims(i_image, -1)
            return cls.get_blks(i_image=image, i_mask=i_mask, i_blk_sizes=i_blk_sizes, i_blk_strides=i_blk_strides)
        else:
            pass
        if i_mask is None:
            i_mask = np.zeros_like(i_image)
        else:
            assert isinstance(i_mask, np.ndarray), '{}'.format(type(i_mask))
            assert len(i_mask.shape) in (2, 3), '{}'.format(i_mask.shape)
            if len(i_mask.shape) == 2:  # For the case of gray mask with shape (height, width)
                mask = np.expand_dims(i_mask, -1)
                return cls.get_blks(i_image=i_image, i_mask=mask, i_blk_sizes=i_blk_sizes,i_blk_strides=i_blk_strides)
            else:
                pass
        ishape = i_image.shape
        mshape = i_mask.shape
        assert ishape[0] == mshape[0], '{} vs {}'.format(ishape[0], mshape[0])
        assert ishape[1] == mshape[1], '{} vs {}'.format(ishape[1], mshape[1])
        if isinstance(i_blk_sizes, int):
            blk_height = i_blk_sizes
            blk_width = i_blk_sizes
        else:
            assert isinstance(i_blk_sizes, (list, tuple))
            assert len(i_blk_sizes) == 2
            blk_height, blk_width = i_blk_sizes
        if isinstance(i_blk_strides, int):
            stride_height = i_blk_strides
            stride_width = i_blk_strides
        else:
            assert isinstance(i_blk_strides, (list, tuple))
            assert len(i_blk_strides) == 2
            stride_height, stride_width = i_blk_strides
        assert isinstance(blk_height, int)
        assert isinstance(blk_width, int)
        assert isinstance(stride_height, int)
        assert isinstance(stride_width, int)
        assert 0 < stride_height <= blk_height, '{} vs {}'.format(stride_height,blk_height)
        assert 0 < stride_width <= blk_width, '{} vs {}'.format(stride_width,blk_width)
        height, width = i_image.shape[0:2]
        blocks, masks = [], []
        for index_h in range(0, height, stride_height):
            row_blocks, row_masks = [], []
            for index_w in range(0, width, stride_width):
                ul_x = index_w
                ul_y = index_h
                br_x = ul_x + blk_width
                br_y = ul_y + blk_height
                if br_x > width or br_y > height:
                    continue
                else:
                    blk = i_image[ul_y:br_y, ul_x:br_x, :]
                    row_blocks.append(blk)
                    mask = i_mask[ul_y:br_y, ul_x:br_x, :]
                    row_masks.append(mask)
            if len(row_blocks) > 0:
                blocks.append(row_blocks)
                masks.append(row_masks)
            else:
                pass
        return blocks, masks
    @classmethod
    def join_blks(cls, i_blks=None, i_steps=None,i_overlapped_adjust=False):
        """Note: Outputs are list of list that indicates the number of blocks in row and cols,respectively"""
        """i_blks[i][j] is an image with shape (m,n,1) or (m,n,3)"""
        """This function must be work with coorperate with the get_blks function above"""
        """i_overlapped_adjust is a special flag that handle the case where the overallaped region has multiple class label"""
        """i_overlapped_adjust is used when joining blocks of mask in segmentation problem with more than 1 objects + 1 background"""
        assert isinstance(i_blks, (list, tuple))
        assert isinstance(i_steps, (list, tuple, int))
        assert isinstance(i_overlapped_adjust,bool)
        if isinstance(i_steps, int):
            stride_height = i_steps
            stride_width  = i_steps
        else:
            assert isinstance(i_steps, (list, tuple))
            assert len(i_steps) == 2
            stride_height, stride_width = i_steps
        num_blk_height = len(i_blks)    # As my design
        num_blk_width  = len(i_blks[0])  # As my design
        blk_height, blk_width = i_blks[0][0].shape[0:2]
        image_height = (num_blk_height - 1) * stride_height + blk_height
        image_width = (num_blk_width - 1) * stride_width + blk_width
        images = []
        mask = None
        for row_index, row in enumerate(i_blks):
            assert isinstance(row, (list, tuple))
            assert len(row) == num_blk_width
            for blk_index, blk in enumerate(row):
                assert isinstance(blk, np.ndarray)
                image = np.zeros(shape=(image_height, image_width, blk.shape[-1]))
                ul_x = blk_index * stride_width
                ul_y = row_index * stride_height
                br_x = ul_x + blk_width
                br_y = ul_y + blk_height
                image[ul_y:br_y, ul_x:br_x] = blk
                if mask is None:
                    mask = np.zeros(shape=(image_height, image_width, blk.shape[-1]), dtype=np.int32)
                else:
                    pass
                mask[ul_y:br_y, ul_x:br_x] = mask[ul_y:br_y, ul_x:br_x] + 1
                images.append(image)
        images = np.array(images)
        if not i_overlapped_adjust:
            images = np.sum(images, axis=0)
            images = (images / mask).astype(np.uint8)
        else:
            """The overalapped region is assigned by the most dominant value"""
            assert len(images.shape)==4
            assert images.shape[-1]==1
            batch,height,width,depth = images.shape
            image = np.zeros(shape=(height, width, 1))
            for h in range(height):
                for w in range(width):
                    vec = images[:,h,w,:]
                    """Find the dominant value in the vec"""
                    max_vec = int(np.max(vec))
                    max_index, max_val = 0, 0
                    for level in range(1,max_vec+1):
                        sum_val = np.sum(vec==level)
                        if sum_val>max_val:
                            max_val   = sum_val
                            max_index = level
                        else:
                            pass
                    image[h,w,:]=max_index
            images = image.copy()
        return images
    @classmethod
    def forward_block_convert(cls,i_blocks):
        """i_blocks is a list of list that contain a (m,n,c) ndarray (images) result of get_block function"""
        """Return: The flatten of i_blocks i.e. list of (m,n,c) ndarray (images)"""
        return [i_blocks[i][j] for i in range(len(i_blocks)) for j in range(len(i_blocks[0]))]
    @classmethod
    def backward_block_convert(cls,i_blocks=None,i_height=1,i_width=1):
        assert isinstance(i_blocks,(list,tuple,np.ndarray))
        if isinstance(i_blocks,np.ndarray):
            blocks = []
            for block in i_blocks:
                blocks.append(block)
            return cls.backward_block_convert(i_blocks=blocks,i_height=i_height,i_width=i_width)
        else:
            pass
        num_blocks = len(i_blocks)
        assert num_blocks == i_height*i_width
        rtn = []
        for index in range(i_height):
            row = i_blocks[index*i_width:(index+1)*i_width]
            rtn.append(row)
        return rtn
    @classmethod
    def get_obj_blks(cls,i_image=None,i_mask=None,i_blk_sizes=None,i_object_size=-1):
        rtn_blocks, rtn_masks = [],[]
        min_object_size = max(cls.vseg_object_size,cls.vcls_object_size,i_object_size)
        blocks, masks = cls.get_blks(i_image=i_image,i_mask=i_mask,i_blk_sizes=i_blk_sizes,i_blk_strides=(16,16))
        blocks = cls.forward_block_convert(blocks)
        masks  = cls.forward_block_convert(masks)
        for index, blk in enumerate(blocks):
            mask = masks[index]
            if np.sum(mask)>min_object_size:
                rtn_blocks.append(blk)
                rtn_masks.append(mask)
        return rtn_blocks,rtn_masks
    """Prepare data for training"""
    def pipeline(self,i_record=None,i_cls_flag=True,i_train_flag=True):
        """As my design, i_record is a dictionary of {'image':[],'label':}"""
        assert isinstance(i_record,dict)      #Raw decode data
        assert isinstance(i_cls_flag,bool)    #Flag for indicating this function pipeline data for clsnet or segnet
        assert isinstance(i_train_flag,bool)
        if i_cls_flag:
            return self.clsnet.pipeline(i_record=i_record,i_ori_shape=self.vcls_isize,i_train_flag=i_train_flag)
        else:
            return self.segnet.pipeline(i_record=i_record,i_ori_shape=self.vseg_isize,i_train_flag=i_train_flag)
    def load_data(self,i_db=None,i_cls_flag=True,i_train_flag=True):
        assert isinstance(i_db,(list, tuple))  # List of (image, mask, label) pair => Used for both cls and segnet cases
        assert isinstance(i_cls_flag, bool)
        assert isinstance(i_train_flag, bool)
        i_save_path = os.path.join(self.vckpts, 'tfrecords')
        if not os.path.exists(i_save_path):
            os.makedirs(i_save_path)
        else:
            pass
        if i_cls_flag:#Classification
            if i_train_flag:
                i_save_path = os.path.join(i_save_path, 'cls_train_db.tfrecord')
            else:
                i_save_path = os.path.join(i_save_path, 'cls_val_db.tfrecord')
        else:#Segmentation
            if i_train_flag:
                i_save_path = os.path.join(i_save_path, 'seg_train_db.tfrecord')
            else:
                i_save_path = os.path.join(i_save_path, 'seg_val_db.tfrecord')
        if os.path.exists(i_save_path):
            dataset = TFRecordDB.read(i_save_path,i_original=True)#Set i_original to True to return dictionary
        else:
            dataset = self.prepare_data(i_db=i_db,i_cls_flag=i_cls_flag,i_train_flag=i_train_flag,i_save_path=i_save_path)
        dataset = dataset.map(lambda x:self.pipeline(i_record=x,i_cls_flag=i_cls_flag,i_train_flag=i_train_flag))
        return dataset
    def prepare_data(self, i_db=None, i_cls_flag=True, i_train_flag=True,i_save_path=None):
        assert isinstance(i_db,(list, tuple))  # List of (image, mask) pair => Used for segnet cases
        assert isinstance(i_cls_flag,bool)
        assert isinstance(i_train_flag,bool)
        if i_cls_flag:
            blk_size    = self.vcls_isize
            blk_strides = self.vcls_strides
            object_size = self.vcls_object_size
            threshold   = self.vcls_sgray_level
        else:
            blk_size    = self.vseg_isize
            blk_strides = self.vseg_strides
            object_size = self.vseg_object_size
            threshold   = 0
        positive_blks,negative_blks = [],[]
        tfrecord_size = 100000
        TFRecordDB.lossy = False
        tfwriter = TFRecordDB()
        for index, element in enumerate(i_db):
            image, mask = element                 # As my design
            assert isinstance(image, np.ndarray)  # Image
            assert isinstance(mask, np.ndarray)   # Mask for segmentation
            if len(image.shape)==2: # Gray image with shape (height, width)
                image = np.expand_dims(image,-1)
            else:
                assert len(image.shape)==3
            assert len(mask.shape) in (2, 3)
            if len(mask.shape)==2:
                mask = np.expand_dims(mask,-1)
            else:
                assert len(mask.shape)==3
                assert mask.shape[-1]==1 # Gray image as the meaning of mask
            """Start extracting blocks"""
            blocks, blk_masks = self.get_blks(i_image=image,i_mask=mask,i_blk_sizes=blk_size,i_blk_strides=blk_strides)
            """Flatten blocks. As my design of get and joint blocks funs"""
            blocks    = self.forward_block_convert(blocks)
            blk_masks = self.forward_block_convert(blk_masks)
            mask_size = np.sum((mask>0).astype(np.int))//10
            for blk_ind,blk in enumerate(blocks):
                blk_mask = blk_masks[blk_ind]
                assert isinstance(blk_mask,np.ndarray)
                if i_cls_flag:#Taking blocks for clsnets
                    if np.average(blk) >= threshold:
                        if np.sum(blk_mask)>max(object_size,mask_size):#Count number of object pixels
                            positive_blks.append(blk)
                        elif np.sum(blk_mask)> 0:
                            pass
                        else:
                            negative_blks.append(blk)
                    else:
                        pass
                else:#Taking blocks for segnets
                    if np.sum(blk_mask)>max(object_size,mask_size): #Only taking blocks with objects
                        positive_blks.append(blk)        #Image
                        negative_blks.append(blk_mask)   #Mask
                    else:
                        pass
            """Complement blocks"""
            obj_blocks,obj_masks = self.get_obj_blks(i_image=image,i_mask=mask,i_blk_sizes=blk_size,i_object_size = mask_size)
            for obj_index, obj_blk in enumerate(obj_blocks):
                positive_blks.append(obj_blk)
                if i_cls_flag:
                    pass
                else:
                    negative_blks.append(obj_masks[obj_index])
            print('Images: {} => Additional blocks = {} vs {} => Sizes = P: {} and N: {}'.format(index, len(obj_blocks), len(obj_masks),len(positive_blks),len(negative_blks)))
        """Save data to TFRecordDB"""
        blocks,labels = [],[]
        if i_cls_flag:
            num_positive_blocks = len(positive_blks)
            num_negative_blocks = len(negative_blks)
            num_samples         = max(num_negative_blocks,num_positive_blocks)
            Logs.log('Num Pos = {}, Num Neg = {}, Num Samples = {}'.format(num_positive_blocks, num_negative_blocks,num_samples))
            if i_train_flag:
                min_samples = min(num_negative_blocks,num_positive_blocks)
                ori_neg_indices = [i for i in range(num_negative_blocks)]
                ori_pos_indices = [i for i in range(num_positive_blocks)]
                neg_indices,pos_indices = [],[]
                if min_samples==num_negative_blocks:
                    ratio   = int(num_positive_blocks/num_negative_blocks)
                    remains = num_positive_blocks - ratio*num_negative_blocks
                    remains_indices = [i for i in range(remains)]
                    for ratio_index in range(ratio):
                        neg_indices += ori_neg_indices
                    neg_indices += remains_indices
                    pos_indices = ori_pos_indices
                    print(ratio,len(neg_indices),len(pos_indices))
                else:
                    ratio = int(num_negative_blocks / num_positive_blocks)
                    remains = num_negative_blocks - ratio * num_positive_blocks
                    remains_indices = [i for i in range(remains)]
                    for ratio_index in range(ratio):
                        pos_indices += ori_pos_indices
                    pos_indices += remains_indices
                    neg_indices = ori_neg_indices
                combine_indices = list(zip(neg_indices,pos_indices))
                for combine_item in combine_indices:
                    neg_index = combine_item[0]
                    pos_index = combine_item[1]
                    blocks.append(positive_blks[pos_index])
                    labels.append(1)
                    blocks.append(negative_blks[neg_index])
                    labels.append(0)
            else:
                blocks = positive_blks + negative_blks
                labels = [1 for _ in range(num_positive_blocks)] + [0 for _ in range(num_negative_blocks)]
            num_positives = np.sum(labels)
            num_negatives = len(labels) - num_positives
            num_blocks    = num_negatives + num_positives
            log_path = os.path.split(i_save_path)[0]
            log_path = os.path.join(log_path, 'statistics.txt')
            with open(log_path, 'a+') as file:
                file.writelines('Statistics for classification db\n')
                file.writelines('Num Positive Blocks: {}/{} ~= {}(%)\n'.format(num_positives, num_blocks,num_positives * 100 / num_blocks))
                file.writelines('Num Negative Blocks: {}/{} ~= {}(%)\n'.format(num_negatives, num_blocks,num_negatives * 100 / num_blocks))
                file.writelines('-' * 100)
                file.writelines('\n')
        else:
            blocks = positive_blks #List of unit8 images
            labels = negative_blks #List of uint8 images
            """Writing statistics"""
            num_blocks   = len(blocks)
            total_sizes  = num_blocks * blk_size[0] * blk_size[1]
            labels_bin   = np.array(labels)
            labels_bin   = (labels_bin>0).astype(np.int)
            object_sizes = np.sum(labels_bin)
            non_object_sizes = total_sizes - object_sizes
            object_sizes     = np.array(object_sizes,dtype=np.int64)
            non_object_sizes = np.array(non_object_sizes,dtype=np.int64)
            total_sizes      = np.array(total_sizes,dtype=np.int64)
            log_path = os.path.split(i_save_path)[0]
            log_path = os.path.join(log_path, 'statistics.txt')
            with open(log_path, 'a+') as file:
                file.writelines('-' * 100 + '\n')
                file.writelines('Statistics for segmentation db\n')
                file.writelines('Num blocks = {}\n'.format(num_blocks))
                file.writelines('Ratio Object    = {}/{} ~ {}(%)\n'.format(object_sizes, total_sizes, (100 * object_sizes) / total_sizes))
                file.writelines('Ratio NonObject = {}/{} ~ {}(%)\n'.format(non_object_sizes, total_sizes,(100 * non_object_sizes) / total_sizes))
                file.writelines('\n')
        """"Write to tfrecords"""
        db_fields = {'image': [], 'label': []}
        write_data = list(zip(blocks, labels))
        tfwriter.write(i_n_records=write_data, i_size=tfrecord_size, i_fields=db_fields, i_save_file=i_save_path)
        dataset = TFRecordDB.read(i_save_path,i_original=True)#Set i_original to True to return dictionary
        return dataset
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module is to implement a stacked CNN-Unet to segment small object in images such as brain stroke lession')
    cls_params = SysParams()
    seg_params = SysParams()
    cls_params.vckpts                    = None           # Checkpoint for storing data
    cls_params.vmodel_name               = 'VGG16'        # Model name
    cls_params.vinput_shape              = (128, 128, 1)  # Input image shape
    cls_params.vnum_classes              = 2              # Number of target classes
    cls_params.vtime_steps               = 1              # For time-sequence classification
    cls_params.vlr                       = 0.0001         # Initial Learning rate
    cls_params.vloss                     = 'swCE'         # Name of loss method
    cls_params.vweights                  = (0.45, 0.55)   # For using weighted cross entropy
    cls_params.vnum_epochs               = 15             # Number of training epochs
    cls_params.vbatch_size               = 32             # Size of batch
    cls_params.vdb_repeat                = 1              # Repeat dataset at single learing rate
    cls_params.vcontinue                 = True           # Continue training or not
    cls_params.vdebug                    = False          # Debug flag
    """=============================================================================================================="""
    seg_params.vckpts                    = None           # Checkpoint for storing data
    seg_params.vinput_shape              = (128, 128, 1)  # Input image shape
    seg_params.vnum_classes              = 2              # Number of target classes
    seg_params.vseg_use_bn               = False          # Segnet parameter
    seg_params.vseg_bblock_type          = 'residual'     # Segnet parameter
    seg_params.vseg_short_cut_rule       = 'concat'       # Segnet parameter
    seg_params.vseg_short_cut_manipulate = True           # Segnet parameter
    seg_params.vlr                       = 0.0001         # Initial Learning rate
    seg_params.vloss                     = 'swCE'         # Name of loss method
    seg_params.vweights                  = (0.45, 0.55)   # For using weighted cross entropy
    seg_params.vnum_epochs               = 30              # Number of training epochs
    seg_params.vbatch_size               = 32             # Size of batch
    seg_params.vdb_repeat                = 1              # Repeat dataset at single learing rate
    seg_params.vcare_background          = False          # Consider background as an object or not
    seg_params.vflip_ud                  = True           # Flip up-down in data augmentation
    seg_params.vflip_lr                  = True           # Flip left-right in data augmentation
    seg_params.vcontinue                 = True           # Continue training or not
    seg_params.vdebug                    = False          # Debug flag
    """=============================================================================================================="""
    StackedCnnUNets.vckpts           = None
    StackedCnnUNets.vcls_params      = cls_params
    StackedCnnUNets.vseg_params      = seg_params
    StackedCnnUNets.vcls_isize       = (64, 64)       # Size of block for clsnet
    StackedCnnUNets.vseg_isize       = (64, 64)       # Size ò block for segnet
    StackedCnnUNets.vcls_strides     = (32, 32)       # Stride for taking blocks for clsnet
    StackedCnnUNets.vseg_strides     = (32, 32)       # Stride for taking blocks for segnet
    StackedCnnUNets.vcls_sgray_level = 50             # Threshold for removing dark blocks (Smallest gray level of considered blocks)
    StackedCnnUNets.vcls_object_size = 5              # Threshold for deciding blocks with/without ground-truth object
    StackedCnnUNets.vseg_object_size = 5              # Threshold for deciding blocks with/without ground-truth object
    StackedCnnUNets.vcls_th          = 0              # Threshold for making decision (extension)
    StackedCnnUNets.vcls_lsm_factor  = 0              # Label smoothing factor. A number from 0 to 1
    StackedCnnUNets.vseg_lsm_factor  = 0              # Label smoothing factor. A number from 0 to 1
    StackedCnnUNets.vdebug           = False          # Debug flag
    trainer = StackedCnnUNets()
    """Get sample dataset"""
    ex_train_db, ex_val_db = SupFns.get_sample_db(i_tsize=(256, 256),i_num_train_samples=1000,i_num_val_samples=100)
    train_images, train_masks, train_labels = ex_train_db
    val_images, val_masks, val_labels       = ex_val_db
    trainer.train(i_train_db=list(zip(train_images,train_masks)),i_val_db=list(zip(val_images,val_masks)))
    trainer.eval(i_db=list(zip(train_images,train_masks,train_labels)))
    trainer.eval(i_db=list(zip(val_images,val_masks,val_labels)))
"""=================================================================================================================="""
