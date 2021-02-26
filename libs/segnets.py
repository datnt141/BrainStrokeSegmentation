import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from libs.logs import Logs
from libs.commons import SupFns
from libs.commons import ListTuples
from libs.sysParams import SysParams
from libs.metrics import SegMetrics_2D
from libs.datasets.tfrecords import TFRecordDB
from libs.callbacks import CustomCallback,LearningRateScheduler
"""=================================================================================================================="""
class ImageSegNets:
    vckpts                    = None            # Checkpoint for storing data
    vinput_shape              = (256,256,3)     # Input image shape
    vnum_classes              = 2               # Number of target classes
    vseg_use_bn               = False           # Segnet parameter
    vseg_bblock_type          = 'residual'      # Segnet parameter
    vseg_short_cut_rule       = 'concat'        # Segnet parameter
    vseg_short_cut_manipulate = True            # Segnet parameter
    vlr                       = 0.0001          # Initial Learning rate
    vloss                     = 'swCE'          # Name of loss method
    vweights                  = (0.45, 0.55)    # For using weighted cross entropy
    vnum_epochs               = 100             # Number of training epochs
    vbatch_size               = 32              # Size of batch
    vdb_repeat                = 1               # Repeat dataset at single learing rate
    vlsm_factor               = 0               # Label smoothing factor
    vtfrecord_size            = 1000            # Number of element of an single tfrecord file
    vcare_background          = False           # Consider background as an object or not
    vcontinue                 = True            # Continue training or not
    vflip_ud                  = True            # Flip up-down in data augmentation
    vflip_lr                  = True            # Flip left-right in data augmentation
    vdebug                    = True            # Debug flag
    def __init__(self):
        """1.Process check point"""
        if self.vckpts is None:
            self.vckpts = os.path.join(os.getcwd(), 'ckpts')
        else:
            assert isinstance(self.vckpts, str), 'Got type: {}'.format(type(self.vckpts))
        if not os.path.exists(self.vckpts):
            os.makedirs(self.vckpts)
        else:
            pass
        """2. Process model path"""
        self.model_path = os.path.join(self.vckpts, 'seg_{}_{}_{}_{}_{}.h5'.format(int(self.vseg_use_bn),self.vseg_bblock_type,self.vseg_short_cut_rule,int(self.vseg_short_cut_manipulate),self.vloss))
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path,custom_objects=SegNets.get_custom_objects())
        else:
            self.model = None
        """3.Init number of filters"""
        if self.vinput_shape[0]   ==  64:
            self.filters = (16,32,64,128)
        elif self.vinput_shape[0] == 128:
            self.filters = (32,64,128,256)
        elif self.vinput_shape[0] == 256:
            self.filters = (16,32,64,128,256)
        elif self.vinput_shape[0] == 512:
            self.filters = (16,32,64,128,256)
        elif self.vinput_shape[0] == 1024:
            self.filters = (16,32,64,128,256,512)
        else:
            raise Exception('Invalid blk size')
        self.mask_shape = (self.vinput_shape[0],self.vinput_shape[1],1) #Always gray image with shape (height, width, 1)
        """4. Assertion"""
        assert isinstance(self.vinput_shape,(list,tuple))
        assert len(self.vinput_shape)==3,'Got value: {}'.format(len(self.vinput_shape))
        assert isinstance(self.vseg_use_bn,bool),'Got type: {}'.format(type(self.vseg_use_bn))
        assert isinstance(self.vseg_bblock_type,str),'Got type: {}'.format(type(self.vseg_bblock_type))
        assert self.vseg_bblock_type in ('norm','residual'),'Got value: {}'.format(self.vseg_bblock_type)
        assert isinstance(self.vseg_short_cut_rule,str),'Got type: {}'.format(type(self.vseg_short_cut_rule))
        assert self.vseg_short_cut_rule in ('none','add','concat'),'Got value: {}'.format(self.vseg_short_cut_rule)
        assert isinstance(self.vseg_short_cut_manipulate,bool),'Got type: {}'.format(type(self.vseg_short_cut_manipulate))
        assert isinstance(self.vloss,str),'Got type: {}'.format(type(self.vloss))
        assert self.vloss in ('wCE', 'swCE', 'FL', 'CE','Dice','swCEnDice','FLnDice'),'Got value: {}'.format(self.vloss)
        assert isinstance(self.vweights,(list,tuple)),'Got type: {}'.format(type(self.vweights))
        assert len(self.vweights)==self.vnum_classes,'Got value: {} vs {}'.format(len(self.vweights),self.vnum_classes)
        assert isinstance(self.vcontinue,bool),'Got type: {}'.format(type(self.vcontinue))
        assert isinstance(self.vnum_classes,int),'Got type: {}'.format(type(self.vnum_classes))
        assert isinstance(self.vnum_epochs,int),'Got type: {}'.format(type(self.vnum_epochs))
        assert isinstance(self.vbatch_size,int),'Got type: {}'.format(type(self.vbatch_size))
        assert isinstance(self.vdb_repeat,int),'Got type: {}'.format(type(self.vdb_repeat))
    def init_params(self,i_params):
        assert isinstance(i_params, (SysParams,dict)), 'Got type: {}'.format(type(i_params))
        Logs.log('-' * 100)
        Logs.log('Init segnet class global variables...')
        """As my design, all global variables of a class start with 'v' letter """
        global_variables = [var for var in self.__dict__ if var.startswith('v')]  # As my design
        if isinstance(i_params, SysParams):
            params = i_params.__dict__
        else:
            params = i_params.copy()
        for key in params.keys():
            val = params[key]
            if key in global_variables:
                Logs.log('Variable {} was changed from {} to : {}'.format(key, self.__dict__[key], val))
                self.__dict__.update({key: val})
            else:
                assert isinstance(key, str),'Got type: {}'.format(type(key))
                if key.startswith('v'):
                    self.__dict__.update({key: val})
                    Logs.log('Init {} as : {}'.format(key, val))
                else:
                    pass
        """Custom parameter adjustment"""
        self.__init__()
        Logs.log('Set-up parameters for segmentation nets:')
        Logs.log_cls_params(self)
        return True
    def train(self,i_train_db=None,i_val_db=None):
        """i_train_db is a tfrecord dataset that contain (image,mask) pair"""
        """i_val_db is similar to i_train_db (or None) that is used for network performance evaluation"""
        if i_train_db is None:#Only valid if we alredy save data to the tfrecord dataset before.
            i_train_db = self.load_data(i_db=i_train_db,i_train_flag=True)
            assert isinstance(i_train_db,tf.data.Dataset),'Got type: {}'.format(type(i_train_db))
        else:
            pass
        assert isinstance(i_train_db,(list,tuple,tf.data.Dataset)),'Got type: {}'.format(type(i_train_db))
        if isinstance(i_train_db,(list,tuple)):
            train_db = self.load_data(i_db=i_train_db,i_train_flag=True)
            if i_val_db is None:
                val_db = None
            else:
                if isinstance(i_val_db,(list,tuple)):
                    val_db = self.load_data(i_db=i_val_db,i_train_flag=False)
                else:
                    assert isinstance(i_val_db,tf.data.Dataset),'Got type: {}'.format(type(i_val_db))
                    val_db = i_val_db
            return self.train(i_train_db=train_db,i_val_db=val_db)
        else:
            assert isinstance(i_train_db,tf.data.Dataset),'Got type: {}'.format(type(i_train_db))
            train_db = i_train_db.batch(self.vbatch_size)
            if i_val_db is None:
                val_db = None
            else:
                if isinstance(i_val_db,(list,tuple)):
                    val_db = self.load_data(i_db=i_val_db,i_train_flag=False)
                    val_db = val_db.batch(self.vbatch_size)
                else:
                    val_db = i_val_db.batch(self.vbatch_size)
        """Start training using tfrecord dataset"""
        if os.path.exists(self.model_path):
            if self.vcontinue:
                Logs.log('Continue training segnet...')
                net = tf.keras.models.load_model(self.model_path,custom_objects=SegNets.get_custom_objects())
            else:
                return False
        else:
            Logs.log('Train model from scratch!')
            """init network"""
            SegNets.seg_use_bn               = self.vseg_use_bn
            SegNets.seg_bblock_type          = self.vseg_bblock_type
            SegNets.seg_short_cut_rule       = self.vseg_short_cut_rule
            SegNets.seg_short_cut_manipulate = self.vseg_short_cut_manipulate
            net = SegNets().build(i_input_shape=self.vinput_shape,i_filters=self.filters, i_num_labels=self.vnum_classes)
        net.summary(print_fn=Logs.log)
        net = SegNets.compile(i_net=net, i_lr=self.vlr, i_loss_name=self.vloss,i_weights=self.vweights)
        """Debugging"""
        if self.vdebug:
            cnt      = 0
            num_test = 5
            for batch in train_db:
                dimages, dlabels = batch
                for dindex, dimage in enumerate(dimages):
                    dlabel = dlabels[dindex]
                    plt.subplot(1,2,1)
                    plt.imshow(dimage,cmap='gray')
                    plt.title('Image')
                    plt.subplot(1,2,2)
                    plt.imshow(tf.argmax(dlabel,axis=-1),cmap='gray')
                    plt.title('Mask')
                    plt.show()
                    cnt += 1
                    if cnt > num_test:
                        break
                    else:
                        pass
                    print('debug_count = ', cnt)
                if cnt > num_test:
                    break
                else:
                    pass
        else:
            pass
        """Training"""
        log_infor  = CustomCallback(i_model_path=self.model_path)
        lr_schuler = LearningRateScheduler()
        lr_params  = {'decay_rule': 1, 'step': int(self.vnum_epochs / 10), 'decay_rate': 0.90, 'base_lr': self.vlr}
        schedule   = lr_schuler(lr_params)
        callbacks  = [schedule, log_infor]
        net.fit(x               = train_db.repeat(self.vdb_repeat),
                epochs          = self.vnum_epochs,
                verbose         = 1,
                shuffle         = True,
                validation_data = val_db,
                callbacks       = callbacks)
        """Update the nework"""
        self.model = tf.keras.models.load_model(self.model_path,custom_objects=SegNets.get_custom_objects())
        return net
    def eval(self,i_db=None):
        """i_db is single-batch image (each element contains only one image and its correponding mask)"""
        assert isinstance(i_db, (list, tuple, tf.data.Dataset)), 'Got type: {}'.format(type(i_db))
        labels, preds = [], []
        for index, element in enumerate(i_db):#Processing for every single image
            print('(SegNets) Evaluating index = {}'.format(index))
            assert isinstance(element, (list, tuple, dict))
            if isinstance(element, (list, tuple)):
                image = element[0]        # As my design. Shape = (None, height, width, nchannels)
                mask  = element[1]        # As my design. Shape = (None, height, width, num_classes)
            else:
                image = element['image']  # As my design. Shape = (None, height, width, nchannels)
                mask  = element['label']  # As my design. Shape = (None, height, width, num_classes)
            if isinstance(image,(tf.Tensor,tf.SparseTensor)):
                image = image.numpy()
            else:
                assert isinstance(image,np.ndarray)
            if isinstance(mask,(tf.Tensor,tf.SparseTensor)):
                mask = mask.numpy()
            else:
                assert isinstance(mask,np.ndarray)
            """Preprocess data"""
            assert len(image.shape) in (2,3,4),'Got shape: {}'.format(image.shape)
            if len(image.shape) in (2,3):#Single image
                """Preprocess image"""
                if len(image.shape)==2:#Gray image with shape (height,width)
                    image = np.expand_dims(image,axis=-1) # Shape: (height, width, 1)
                else:#Shape: (height, width, depth)
                    pass
                assert len(image.shape) == 3, 'Got shape: {}'.format(image.shape)
                assert image.shape[-1] in (1, 3), 'Got shape: {}'.format(image.shape)
                """Preprocess mask"""
                assert len(mask.shape) in (2, 3)
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, -1)
                else:
                    assert mask.shape[-1] == 1 #Only gray image with shape (height, width, 1)
                """Making batch for single image"""
                image = np.expand_dims(image,0)
                mask  = np.expand_dims(mask,0)
            else:#Batch of images
                assert len(image.shape)==4 #Shape: (None, height, width, depth)
                assert len(mask.shape) in (3,4)
                if len(mask.shape)==3:
                    mask = np.expand_dims(mask,axis=-1)
                else:
                    assert mask.shape[-1] == 1, 'Got shape: {}'.format(mask.shape)
            assert len(image.shape)==4, 'Got shape: {}'.format(image.shape) #Shape: (None, height, width, depth)
            assert len(mask.shape)==4, 'Got shape: {}'.format(mask.shape)   #Shape: (None, height, width, 1)
            assert image.shape[-1] in (1, 3), 'Got shape: {}'.format(image.shape)
            assert mask.shape[-1] == 1, 'Got shape: {}'.format(mask.shape)
            cpreds = self.predict(i_image=image)    # Shape = (1, height, width, 1) as procesisng single image
            for pindex, cpred in enumerate(cpreds):
                cmask = mask[pindex]
                if np.sum(ListTuples.compare(i_x=cmask.shape,i_y=self.mask_shape)):
                    cmask = SupFns.scale_mask(i_mask=cmask,i_tsize=self.mask_shape)
                else:
                    pass
                labels.append(cmask)   # Shape = (height, width, 1)
                preds.append(cpred)    # Shape = (height, width, 1)
                print(labels[-1].shape,preds[-1].shape)
                if self.vdebug:
                    plt.subplot(1,3,1)
                    plt.imshow(image[pindex],cmap='gray')
                    plt.title('Original Image')
                    plt.subplot(1,3,2)
                    plt.imshow(mask[pindex],cmap='gray')
                    plt.title('Mask')
                    plt.subplot(1,3,3)
                    plt.imshow(cpred,cmap='gray')
                    plt.title('Prediction')
                    plt.show()
                else:
                    pass
        """Performance measurement"""
        evaluer = SegMetrics_2D(i_num_classes=self.vnum_classes,i_care_background=self.vcare_background)
        Logs.log('Using entire dataset')
        measures, measure_mean, measure_std = evaluer.eval(i_labels=labels, i_preds=preds,i_object_care=False)
        Logs.log('Measure shape = {}'.format(measures.shape))
        Logs.log('Measure mean  = {}'.format(measure_mean))
        Logs.log('Measure std   = {}'.format(measure_std))
        Logs.log('Using sub dataset that only consider images containing objects')
        measures, measure_mean, measure_std = evaluer.eval(i_labels=labels, i_preds=preds, i_object_care=True)
        Logs.log('Measure shape = {}'.format(measures.shape))
        Logs.log('Measure mean  = {}'.format(measure_mean))
        Logs.log('Measure std   = {}'.format(measure_std))
        return labels, preds
    def predict(self,i_image=None):
        assert isinstance(self.model,tf.keras.models.Model), 'Got type: {}'.format(type(self.model))
        assert isinstance(i_image,np.ndarray),'Got type: {}'.format(type(i_image))
        assert len(i_image.shape) in (2, 3,4), 'Got value: {}'.format(i_image.shape) #Image or batch of images
        if len(i_image.shape) in (2, 3):#For single image
            if len(i_image.shape) == 2:#Gray image with shape (height, width)
                images = np.expand_dims(i_image,-1)
            else:#Color image with shape (height, width, depth)
                images = i_image.copy()
            assert len(images.shape)==3
            images = np.expand_dims(i_image,0)
        else:#For batch of image
            images = i_image.copy()
        assert len(images.shape)==4, 'Got shape: {}'.format(images.shape)
        assert images.shape[-1] in (1,3),'Got shape: {}'.format(images.shape)
        """Size Normalization"""
        nimages = []
        for image in images:
            assert isinstance(image,np.ndarray), 'Got type: {}'.format(type(image))
            """Color adjustment"""
            if image.shape[-1]==self.vinput_shape[-1]:
                pass
            else:
                if image.shape[-1]==1:#Convert to color image
                    image = np.concatenate((image,image,image),axis=-1)
                else:#Convert to gray image
                    image = (np.average(image,axis=-1)).astype(np.uint8)
            """Resizing"""
            nimages.append(SupFns.imresize(i_image=image,i_tsize=self.vinput_shape[0:2]))
        images = np.array(nimages)
        assert len(images.shape)==4, 'Got shape: {} with length = {}'.format(images.shape,len(images.shape))
        assert images.shape[-1] in (1, 3)
        assert images.dtype in (np.uint8,) #As it is result of SupFns.imresize()
        """Normalization"""
        images = images/255.0
        """Prediction"""
        pred = self.model.predict(images)                       #Return: (None, height, width, num_classes)
        return np.expand_dims(np.argmax(pred,axis=-1),axis=-1)  #Return: (None, height, width,1)
    """Data preparation"""
    def pipeline(self,i_record=None, i_ori_shape=None, i_train_flag=True):
        """i_record is the output of tf.data.Dataset after doing pipeline()"""
        """i_ori_shape is the original shape of input image"""
        assert isinstance(i_record, (list, tuple, dict)), 'Got type: {}'.format(type(i_record))
        assert isinstance(i_ori_shape, (list, tuple)), 'Got type: {}'.format(type(i_ori_shape))
        assert isinstance(i_train_flag,bool),'Got type: {}'.format(type(i_train_flag))
        assert len(i_ori_shape) in (2, 3),'Got shape: {} with lenght = {}'.format(i_ori_shape, len(i_ori_shape))
        if len(i_ori_shape) == 2:
            i_ori_shape = (i_ori_shape[0], i_ori_shape[1], 1)
        else:
            assert i_ori_shape[-1] in (1, 3), 'Got value: {}'.format(i_ori_shape)
        assert 0<i_ori_shape[0]<=i_ori_shape[1], 'Got value: {}'.format(i_ori_shape)
        """Init the original shape of mask"""
        ori_mask_shape = (i_ori_shape[0], i_ori_shape[1], 1)
        if isinstance(i_record, (list, tuple)):
            image = i_record[0] #Uint8 image
            mask  = i_record[1] #Integer mask
        else:
            image = i_record['image'] #As my design. An uint8 image
            mask  = i_record['label'] #As my design. Consider change 'label' to 'mask'
        assert isinstance(image, (tf.Tensor, tf.SparseTensor)), 'Got type: {}'.format(type(image))
        assert isinstance(mask,(tf.Tensor,tf.SparseTensor)),'Got type: {}'.format(type(mask))
        assert image.dtype in (tf.dtypes.uint8,),'Got dtype: {}'.format(image.dtype)
        assert mask.dtype in (tf.dtypes.uint8,),'Got dtype: {}'.format(mask.dtype)
        image = tf.reshape(tensor=image,shape=i_ori_shape)
        mask  = tf.reshape(tensor=mask,shape=ori_mask_shape)
        """Processing image and label"""
        """Color adjustment"""
        if i_ori_shape[-1] == self.vinput_shape[-1]:
            pass
        else:
            if i_ori_shape[-1]==1:#RGB image
                image = tf.concat(values=(image,image,image),axis=-1)
            else:#Gray image
                image = tf.reshape(tf.cast(tf.reduce_mean(input_tensor=image,axis=-1),tf.dtypes.uint8),shape=(i_ori_shape[0],i_ori_shape[1],1))
        """Scaling"""
        image = tf.cast(tf.image.resize(image, size=(self.vinput_shape[0],self.vinput_shape[1])),tf.dtypes.uint8)
        """Scaling mask with individual object index to ensure label index"""
        masks = [tf.zeros(shape=(self.vinput_shape[0],self.vinput_shape[1],1),dtype=tf.dtypes.uint8)]
        for index in range(1,self.vnum_classes):
            """Taking mask of index(th) label component"""
            cmask = tf.cast(tf.where(tf.equal(mask,index),tf.ones_like(mask),tf.zeros_like(mask)),tf.dtypes.uint8) #Shape: (None, height, width, 1)
            """Resizing"""
            cmask = tf.cast(tf.image.resize(tf.cast(tf.multiply(cmask,255),tf.dtypes.uint8), size=(self.vinput_shape[0],self.vinput_shape[1])), tf.uint8)
            """Thresholding"""
            cmask = tf.cast(tf.where(tf.greater(cmask,0),tf.ones_like(cmask),tf.zeros_like(cmask)),tf.dtypes.uint8)
            """Reconstructing label"""
            cmask = tf.cast(tf.multiply(cmask,index),tf.dtypes.uint8)
            masks.append(cmask)
        mask  = tf.concat(masks,axis=-1)
        mask  = tf.reduce_max(mask,axis=-1) #Selecting the object with max index
        """One-hot encodeing with/without label smoothing again"""
        image = tf.reshape(tensor=tf.cast(image,tf.dtypes.uint8), shape=self.vinput_shape)
        mask  = tf.reshape(tensor=tf.cast(mask,tf.dtypes.uint8), shape=self.mask_shape)
        if i_train_flag:
            images = tf.concat(values=(image, mask), axis=-1)
            if self.vflip_ud:
                images = tf.image.random_flip_up_down(images)
            else:
                pass
            if self.vflip_lr:
                images = tf.image.random_flip_left_right(images)
            else:
                pass
            image, mask = tf.split(value=images, num_or_size_splits=[self.vinput_shape[-1], self.mask_shape[-1]],axis=-1)
            """Normalization"""
            image = tf.reshape(tensor=tf.cast(image, tf.dtypes.float32), shape=self.vinput_shape) / 255.0
            mask  = tf.cast(mask, tf.dtypes.uint8)
            mask  = tf.one_hot(indices=tf.squeeze(tf.reshape(tensor=mask, shape=self.mask_shape)),depth=self.vnum_classes)  # For using categoricalloss
            """Label smoothing"""
            mask = mask * (1.0 - self.vlsm_factor)
            mask = mask + self.vlsm_factor / self.vnum_classes
        else:
            """Normalization"""
            image = tf.cast(image,tf.dtypes.float32) / 255.0
            mask  = tf.cast(mask, tf.dtypes.uint8)
            mask  = tf.one_hot(indices=tf.squeeze(tf.reshape(tensor=mask, shape=self.mask_shape)),depth=self.vnum_classes)  # For using categoricalloss
        return image, mask
    def load_data(self,i_db=None,i_train_flag=True):
        """This function is to prepare data for training or testing using conventional list of images"""
        """As my design, i_db is list of (image,label) or (image_path,label)"""
        assert isinstance(i_train_flag,bool), 'Got type: {}'.format(type(i_train_flag))
        i_save_path = os.path.join(self.vckpts, 'tfrecords')
        if not os.path.exists(i_save_path):
            os.makedirs(i_save_path)
        else:
            pass
        if i_train_flag:
            i_save_path = os.path.join(i_save_path,'seg_train_db.tfrecord')
        else:
            i_save_path = os.path.join(i_save_path,'seg_val_db.tfrecord')
        if os.path.exists(i_save_path):
            dataset = TFRecordDB.read(i_tfrecord_path=i_save_path,i_original=True) #Set i_original to True to return dictionary
        else:
            assert isinstance(i_db, (list, tuple)), 'Got type: {}'.format(type(i_db))
            dataset  = self.prepare_db(i_db=i_db,i_save_path=i_save_path)
        dataset = dataset.map(lambda x:self.pipeline(i_record=x,i_ori_shape=self.vinput_shape,i_train_flag=i_train_flag))
        return dataset
    """Prepare tf.data.Dataset object. DONOT USE THIS FUNCTION DIRECTLY. USE load_data() instead"""
    def prepare_db(self,i_db=None,i_save_path=None):
        """i_db is a list (tuple) of (image, label) or (image_path,label_path) or (image,label_path) or (image_path, label)"""
        assert isinstance(i_db,(list,tuple)), 'Got type: {}'.format(type(i_db))
        assert isinstance(i_save_path,str), 'Got type: {}'.format(type(i_save_path))
        assert not os.path.exists(i_save_path), 'Got path: {}'.format(i_save_path)
        save_path, save_file = os.path.split(i_save_path)
        save_file      = save_file[0:len(save_file) - len('.tfrecord')]
        db_fields      = {'image': [], 'label': []}
        images, labels = [], []
        count          = 0
        segment        = 0
        TFRecordDB.lossy = False
        tfwriter         = TFRecordDB()
        for index,element in enumerate(i_db):
            image,mask = element #As my design
            """Load image"""
            if isinstance(image,str):
                image = imageio.imread(image)
            else:
                assert isinstance(image,np.ndarray), 'Got type: {}'.format(type(image))
            if isinstance(mask,str):
                mask  = imageio.imread(mask)
            else:
                assert isinstance(mask,np.ndarray), 'Got type: {}'.format(type(mask))
            """Validate the image and mask"""
            assert len(image.shape) in (2,3), 'Got shape: {}'.format(image.shape)
            if len(image.shape)==2:
                image = np.expand_dims(image,-1)
            else:
                assert image.shape[-1] in (1,3), 'Got shape: {}'.format(image.shape)
            assert len(mask.shape) in (2,3), 'Got shape: {}'.format(mask.shape)
            if len(mask.shape)==2:
                mask = np.expand_dims(mask,-1)
            else:
                assert mask.shape[-1] in (1,), 'Got shape: {}'.format(mask.shape)
            """Color adjustment"""
            if image.shape[-1] == self.vinput_shape[-1]:
                pass
            else:
                assert self.vinput_shape[-1] in (1, 3)
                if image.shape[-1]==1:
                    image = np.concatenate((image,image,image),axis=-1)
                else:
                    image = (np.average(image,axis=-1)).astype(np.uint8)
            """Scaling if needed"""
            if np.sum(ListTuples.compare(i_x=image.shape,i_y=self.vinput_shape)):
                image = SupFns.imresize(i_image=image, i_tsize=self.vinput_shape[0:2])
            else:
                pass
            if np.sum(ListTuples.compare(i_x=mask.shape,i_y=self.mask_shape)):
               mask  = SupFns.scale_mask(i_mask=mask,i_tsize=self.vinput_shape[0:2])
            else:
                pass
            assert isinstance(image,np.ndarray), 'Got type: {}'.format(type(image))
            assert isinstance(mask,np.ndarray),'Got type: {}'.format(type(mask))
            assert image.dtype in (np.uint8,), 'Got dtype: {}'.format(image.dtype)
            assert mask.dtype in(np.uint8,),'Got dtype: {}'.format(mask.dtype)
            images.append(image)
            labels.append(mask)
            count +=1
            if count==self.vtfrecord_size:
                if segment==0:
                    csave_path = os.path.join(save_path,'{}.tfrecord'.format(save_file))
                else:
                    csave_path = os.path.join(save_path,'{}_{}.tfrecord'.format(save_file,segment))
                write_data = list(zip(images, labels))
                tfwriter.write(i_n_records=write_data, i_size=self.vtfrecord_size, i_fields=db_fields, i_save_file=csave_path)
                """Clear images and labels lists"""
                count = 0
                images.clear()
                labels.clear()
                segment +=1
            else:
                pass
        """Write the final part"""
        if len(images) > 0:
            if segment == 0:
                csave_path = os.path.join(save_path,'{}.tfrecord'.format(save_file))
            else:
                csave_path = os.path.join(save_path, '{}_{}.tfrecord'.format(save_file, segment))
            write_data = list(zip(images, labels))
            tfwriter.write(i_n_records=write_data, i_size=self.vtfrecord_size, i_fields=db_fields,i_save_file=csave_path)
        else:
            pass
        dataset = tfwriter.read(i_tfrecord_path=i_save_path, i_original=True)
        return dataset
"""=================================================================================================================="""
"""=================================================================================================================="""
class SegNets:
    seg_bblock_type          = 'residual'  # Swith among 'norm','residual'
    seg_short_cut_rule       = 'concat'      # Switch among 'none','add','concat'
    seg_short_cut_manipulate = True
    seg_use_bn               = False
    def __init__(self):
        pass
    @classmethod
    def bblock(cls,i_inputs=None,i_nb_filters=32):
        assert isinstance(i_nb_filters, int)
        assert i_nb_filters > 0
        if cls.seg_bblock_type == 'norm':
            return cls.conv(i_inputs=i_inputs, i_kernel_size=3,i_nb_filters=(i_nb_filters, 2*i_nb_filters, i_nb_filters))
        elif cls.seg_bblock_type == 'residual':
            return cls.residual(i_inputs=i_inputs, i_kernel_size=3,i_nb_filters=(i_nb_filters,2*i_nb_filters,i_nb_filters))
        else:
            raise Exception('Invalid building block name')
    """Main buiding function"""
    def build(self,i_input_shape= (256,256,3),i_filters=(16,32,64,128,256),i_num_labels=2):
        """Unet model in details"""
        assert isinstance(i_filters, tuple)
        assert len(i_filters) > 0
        assert isinstance(i_num_labels, int)
        assert i_num_labels > 0
        i_filters = list(i_filters)
        """1. Input layers"""
        inputs = tf.keras.layers.Input(shape=i_input_shape)
        """2. Warm-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters[0],kernel_size=(7,7),strides=(1,1),activation='relu',padding='same')(inputs)
        skips = []
        """3. Encoder part"""
        for nb_filter in i_filters:
            outputs = self.bblock(i_inputs=outputs, i_nb_filters=nb_filter)
            skips.append(outputs)
            outputs = tf.keras.layers.Conv2D(filters=nb_filter,kernel_size=3,strides=(2,2),padding='same')(outputs)
        outputs = self.bblock(i_inputs=outputs, i_nb_filters=i_filters[-1])
        """4. Decoder part"""
        i_filters.reverse()
        skips.reverse()
        for index, nb_filter in enumerate(i_filters):
            outputs = tf.keras.layers.UpSampling2D(size=(2, 2))(outputs)
            if self.seg_short_cut_rule == 'none':  # Donot use short cut path between encoder-decoder
                pass
            elif self.seg_short_cut_rule == 'concat':  # Concatenate short-cut path between encoder-decoder
                if self.seg_short_cut_manipulate:
                    skip_net = self.bblock(i_inputs=skips[index], i_nb_filters=skips[index].shape[-1])
                else:
                    skip_net = skips[index]
                outputs = tf.keras.layers.Concatenate(axis=-1)([outputs, skip_net])
            elif self.seg_short_cut_rule == 'add':  # Addition of short-cut and feature maps between encoder-decoder
                if self.seg_short_cut_manipulate:
                    skip_net = self.bblock(i_inputs=skips[index], i_nb_filters=skips[index].shape[-1])
                else:
                    skip_net = skips[index]
                outputs = tf.keras.layers.Add()([outputs, skip_net])
            else:
                raise Exception('Invalid short cut rule!')
            outputs = self.bblock(i_inputs=outputs, i_nb_filters=nb_filter)
            if index == len(i_filters) - 1:
                index = len(i_filters) - 2
            else:
                pass
            outputs = self.bblock(i_inputs=outputs, i_nb_filters=i_filters[index + 1])
        """4. Output"""
        outputs = tf.keras.layers.Conv2D(filters=i_num_labels, kernel_size=(3, 3), strides=(1, 1), padding="same",name='output')(outputs)
        model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
        return model
    def build_attention(self,i_input_shape= (256,256,3),i_filters=(16,32,64,128,256),i_num_labels=2):
        """Unet model in details"""
        assert isinstance(i_filters, tuple)
        assert len(i_filters) > 0
        assert isinstance(i_num_labels, int)
        assert i_num_labels > 0
        i_filters = list(i_filters)
        """1. Input layers"""
        inputs_a = tf.keras.layers.Input(shape=i_input_shape)
        inputs_b = tf.keras.layers.Input(shape=(i_input_shape[0],i_input_shape[1],1))
        """2. Warm-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters[0], kernel_size=(7, 7), strides=(1, 1), activation='relu',padding='same')(inputs_a)
        skips   = []
        att_tensors = inputs_b
        """3. Encoder part"""
        height, width = i_input_shape[0:2]
        for nb_filter in i_filters:
            outputs = self.bblock(i_inputs=outputs, i_nb_filters=nb_filter)
            skips.append(outputs)
            """Post processing here"""
            outputs = tf.keras.layers.Conv2D(filters=nb_filter, kernel_size=3, strides=(1, 1), padding='same')(outputs)
            print('1 ',outputs.shape)
            """Attention block adding here"""
            height = height // 2
            width = width // 2
            att_tensors = tf.image.resize(images=att_tensors, size=(height, width))
            print('2 ', att_tensors.shape)
            outputs     = self.attention_block(i_inputs=outputs, i_att_inputs=att_tensors, i_nb_filters=nb_filter)
            print('3 ',outputs.shape)
        outputs = self.bblock(i_inputs=outputs, i_nb_filters=i_filters[-1])
        """4. Decoder part"""
        i_filters.reverse()
        skips.reverse()
        for index, nb_filter in enumerate(i_filters):
            outputs = tf.keras.layers.UpSampling2D(size=(2, 2))(outputs)
            if self.seg_short_cut_rule == 'none':  # Donot use short cut path between encoder-decoder
                pass
            elif self.seg_short_cut_rule == 'concat':  # Concatenate short-cut path between encoder-decoder
                if self.seg_short_cut_manipulate:
                    skip_net = self.bblock(i_inputs=skips[index], i_nb_filters=skips[index].shape[-1])
                else:
                    skip_net = skips[index]
                outputs = tf.keras.layers.Concatenate(axis=-1)([outputs, skip_net])
            elif self.seg_short_cut_rule == 'add':  # Addition of short-cut and feature maps between encoder-decoder
                if self.seg_short_cut_manipulate:
                    skip_net = self.bblock(i_inputs=skips[index], i_nb_filters=skips[index].shape[-1])
                else:
                    skip_net = skips[index]
                outputs = tf.keras.layers.Add()([outputs, skip_net])
            else:
                raise Exception('Invalid short cut rule!')
            outputs = self.bblock(i_inputs=outputs, i_nb_filters=nb_filter)
            if index == len(i_filters) - 1:
                index = len(i_filters) - 2
            else:
                pass
            outputs = self.bblock(i_inputs=outputs, i_nb_filters=i_filters[index + 1])
        """4. Output"""
        outputs = tf.keras.layers.Conv2D(filters=i_num_labels, kernel_size=(3, 3), strides=(1, 1), padding="same",name='output')(outputs)
        model = tf.keras.models.Model(inputs=[inputs_a,inputs_b], outputs=[outputs])
        return model
    """Support functions"""
    @staticmethod
    def attention_block(i_inputs, i_att_inputs, i_nb_filters=32):
        """Perform attention block"""
        """Reference: Attention Enriched Deep Learning Model for Brest Tumor Segmentation in Ultrasound Images"""
        pool_outputs = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(i_inputs)
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(i_inputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        att_outputs = tf.keras.layers.Conv2D(filters=i_nb_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(i_att_inputs)
        att_outputs = tf.keras.layers.Activation('relu')(att_outputs)
        outputs = tf.keras.layers.Add()([outputs, att_outputs])
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(outputs)
        outputs = tf.keras.layers.Activation('sigmoid')(outputs)
        outputs = tf.keras.layers.Multiply()([pool_outputs, outputs])
        return outputs
    @classmethod
    def conv(cls, i_inputs, i_kernel_size, i_nb_filters, use_bias=True):
        assert isinstance(i_kernel_size, int)
        assert isinstance(i_nb_filters, (list, tuple))
        assert len(i_nb_filters) == 3
        nb_filter1, nb_filter2, nb_filter3 = i_nb_filters
        """First CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter1, kernel_size=(1, 1), use_bias=use_bias)(i_inputs)
        if cls.seg_use_bn:
            outputs = tf.keras.layers.BatchNormalization()(outputs)
        else:
            pass
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Second CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter2, kernel_size=(i_kernel_size, i_kernel_size), padding='same',use_bias=use_bias)(outputs)
        if cls.seg_use_bn:
            outputs = tf.keras.layers.BatchNormalization()(outputs)
        else:
            pass
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Third CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter3, kernel_size=(1, 1), use_bias=use_bias)(outputs)
        if cls.seg_use_bn:
            outputs = tf.keras.layers.BatchNormalization()(outputs)
        else:
            pass
        outputs = tf.keras.layers.Activation('relu')(outputs)
        return outputs
    @classmethod
    def residual(cls, i_inputs, i_kernel_size, i_nb_filters, i_strides=(1, 1), use_bias=True):
        assert isinstance(i_kernel_size, int)
        assert isinstance(i_nb_filters, (list, tuple))
        assert len(i_nb_filters) == 3
        nb_filter1, nb_filter2, nb_filter3 = i_nb_filters
        """First CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter1, kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias)(
            i_inputs)
        if cls.seg_use_bn:
            outputs = tf.keras.layers.BatchNormalization()(outputs)
        else:
            pass
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Second CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter2, kernel_size=(i_kernel_size, i_kernel_size),strides=i_strides, padding='same', use_bias=use_bias)(outputs)
        if cls.seg_use_bn:
            outputs = tf.keras.layers.BatchNormalization()(outputs)
        else:
            pass
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Third CONV block"""
        outputs = tf.keras.layers.Conv2D(nb_filter3, (1, 1), use_bias=use_bias)(outputs)
        if cls.seg_use_bn:
            outputs = tf.keras.layers.BatchNormalization()(outputs)
        else:
            pass
        """Shortcut CONV block"""
        shortcut = tf.keras.layers.Conv2D(nb_filter3, (1, 1), strides=i_strides, use_bias=use_bias)(i_inputs)
        if cls.seg_use_bn:
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        else:
            pass
        """Aggregation"""
        outputs = tf.keras.layers.Add()([outputs, shortcut])
        outputs = tf.keras.layers.Activation('relu')(outputs)
        return outputs
    @staticmethod
    def weighted_ce(i_weights=None):
        assert isinstance(i_weights,(list,tuple,tf.Tensor,tf.SparseTensor))
        epsilon = tf.keras.backend.epsilon()
        if isinstance(i_weights,(list,tuple)):
            weights = tf.convert_to_tensor(i_weights)
        else:
            weights = tf.cast(i_weights,tf.float32)
        def get_wce(i_y_true,i_y_pred):
            y_true = tf.cast(i_y_true, tf.float32)                    # Result: (None, height, width, num_classes)
            y_pred = tf.nn.softmax(i_y_pred, axis=-1)                 # Result: (None, height, width, num_classes)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Result: (None, height, width, num_classes)
            wce    = -y_true * tf.math.log(y_pred) * weights          # Broast-casting. #Result: (None, height, width, num_classes)
            return tf.reduce_mean(tf.reduce_sum(wce, axis=-1), axis=None)
        return get_wce
    @staticmethod
    def sweighted_ce(i_y_true,i_y_pred):
        """i_y_true is Tensor in shape [None, height, width, num_classes] in form of one-hot encoding"""
        """i_y_pred is Tensor in shape [None, height, width, num_classes] in form of unscaled (original) digits"""
        """tf.nn.sparse_softmax_cross_entropy_with_logits requires unscaled logit"""
        y_true  = tf.cast(i_y_true, tf.float32)
        samples = tf.reduce_sum(y_true,axis=[0,1,2])        #Result: (num_classes,). Number of samples in each classes
        weights = tf.divide(samples,tf.reduce_sum(samples)) #Result: (num_classes,). Summation = 1.
        weights = tf.subtract(1.0,weights)
        return SegNets.weighted_ce(i_weights=weights)(i_y_true=i_y_true,i_y_pred=i_y_pred)
    @staticmethod
    def focal_loss(i_gamma=2.0,i_alpha=0.25):
        epsilon = tf.keras.backend.epsilon()
        def get_fl(i_y_true,i_y_pred):
            """i_y_true is Tensor in shape [None, height, width, num_classes] in form of one-hot encoding"""
            """i_y_pred is Tensor in shape [None, height, width, num_classes] in form of unscaled (original) digits"""
            y_true = tf.cast(i_y_true,tf.float32)                    # Result: (None, height, width, num_classes)
            y_pred = tf.nn.softmax(i_y_pred, axis=-1)                # Result: (None, height, width, num_classes)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon) # Result: (None, height, width, num_classes)
            ce     = -y_true*tf.math.log(y_pred)                     # Result: (None, height, width, num_classes)
            loss   = i_alpha*tf.math.pow(1.0-y_pred,i_gamma)*ce      # Result: (None, height, width, num_classes)
            return tf.reduce_mean(tf.reduce_sum(loss,axis=-1),axis=None)
        return get_fl
    @staticmethod
    def dice_loss(i_y_true,i_y_pred):
        """i_y_true is Tensor in shape [None, height, width, num_classes] in form of one-hot encoding"""
        """i_y_pred is Tensor in shape [None, height, width, num_classes] in form of unscaled (original) digits"""
        epsilon = tf.keras.backend.epsilon()
        """Measurement of numerator"""
        y_true = tf.cast(i_y_true,tf.float32)                     # Result: (None, height, width, num_classes)
        y_pred = tf.nn.softmax(i_y_pred, axis=-1)                 # Result: (None, height, width, num_classes)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Result: (None, height, width, num_classes)
        numerator = y_true * y_pred                               # Result: (None, height, width, num_classes)
        numerator = tf.reduce_sum(numerator,axis=[1,2])           # Result: (None, num_classes)
        """Measurement of denominator"""
        denominator = tf.square(y_pred) + tf.square(y_true)       # Result: (None, height, width, num_classes)
        denominator = tf.reduce_sum(denominator, axis=[1, 2])     # Result: (None, num_classes)
        """Output the results"""
        dice = 1.0 - (2*numerator+epsilon)/(denominator+epsilon)  # Result: (None, num_classes)
        #dice = tf.reduce_sum(dice,axis=-1)                       # Result: (None, )
        #dice = tf.reduce_mean(dice)
        backgournd_dice,foreground_dice = tf.split(dice,num_or_size_splits=[1,-1],axis=-1)
        #print(bdice.shape,fdice.shape,y_true.shape,y_pred.shape)
        dice = tf.reduce_mean(foreground_dice)
        return dice
    @staticmethod
    def swCEwithDice_loss(i_y_true,i_y_pred):
        swCE_loss = SegNets.sweighted_ce(i_y_true=i_y_true,i_y_pred=i_y_pred)
        D_loss    = SegNets.dice_loss(i_y_true=i_y_true,i_y_pred=i_y_pred)
        return swCE_loss + D_loss
    @staticmethod
    def FLwithDice_loss(i_y_true, i_y_pred):
        FL_loss = SegNets.focal_loss(i_gamma=2.0,i_alpha=0.25)(i_y_true=i_y_true, i_y_pred=i_y_pred)
        D_loss  = SegNets.dice_loss(i_y_true=i_y_true, i_y_pred=i_y_pred)
        return FL_loss + D_loss
    @staticmethod
    def compile(i_net=None,i_lr=0.001,i_loss_name='wCE',i_weights=None):
        assert isinstance(i_loss_name,str)
        assert i_loss_name in('wCE','swCE','FL','CE','Dice','swCEnDice','FLnDice')
        assert isinstance(i_weights,(list,tuple,tf.Tensor,tf.SparseTensor))
        if i_loss_name == 'wCE':
            loss  = SegNets.weighted_ce(i_weights=i_weights)
        elif i_loss_name == 'swCE':
            loss  = SegNets.sweighted_ce
        elif i_loss_name == 'FL':
            loss  = SegNets.focal_loss(i_gamma=2.0,i_alpha=0.25)
        elif i_loss_name == 'Dice':
            loss  = SegNets.dice_loss
        elif i_loss_name == 'swCEnDice':
            loss = SegNets.swCEwithDice_loss
        elif i_loss_name == 'FLnDice':
            loss = SegNets.FLwithDice_loss
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        i_net.compile(optimizer  = tf.keras.optimizers.Adam(lr=i_lr),
                      loss       = loss,
                      metrics    = ['accuracy'])
        return i_net
    @staticmethod
    def get_custom_objects():
        custom_objects = {'get_wce'           : SegNets.weighted_ce(i_weights=[0.45,0.55]),
                          'sweighted_ce'      : SegNets.sweighted_ce,
                          'get_fl'            : SegNets.focal_loss(i_gamma=2.,i_alpha=0.25),
                          'dice_loss'         : SegNets.dice_loss,
                          'swCEwithDice_loss' : SegNets.swCEwithDice_loss,
                          'FLwithDice_loss'   : SegNets.FLwithDice_loss}
        return custom_objects
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module is to train segmentation network')
    print('Please set the vckpts carefully as all of code and results will be stored and look at this directory as my design')
    seg_params = SysParams()
    seg_params.vckpts                    = None           # Checkpoint for storing data
    seg_params.vinput_shape              = (128, 128, 3)  # Input image shape
    seg_params.vnum_classes              = 2              # Number of target classes
    seg_params.vseg_use_bn               = False          # Segnet parameter
    seg_params.vseg_bblock_type          = 'residual'     # Segnet parameter
    seg_params.vseg_short_cut_rule       = 'concat'       # Segnet parameter
    seg_params.vseg_short_cut_manipulate = True           # Segnet parameter
    seg_params.vlr                       = 0.0001         # Initial Learning rate
    seg_params.vloss                     = 'swCE'         # Name of loss method
    seg_params.vweights                  = (0.45, 0.55)   # For using weighted cross entropy
    seg_params.vnum_epochs               = 100            # Number of training epochs
    seg_params.vbatch_size               = 8              # Size of batch
    seg_params.vdb_repeat                = 1              # Repeat dataset at single learing rate
    seg_params.vlsm_factor               = 0              # Label smoothing factor
    seg_params.vtfrecord_size            = 1000           # Number of element of an single tfrecord file
    seg_params.vcare_background          = False          # Consider background as an object or not
    seg_params.vcontinue                 = False          # Continue training or not
    seg_params.vflip_ud                  = True           # Flip up-down in data augmentation
    seg_params.vflip_lr                  = True           # Flip left-right in data augmentation
    seg_params.vdebug                    = True           # Debug flag
    trainer = ImageSegNets()
    trainer.init_params(i_params=seg_params)
    """Get sample dataset"""
    ex_train_db, ex_val_db = SupFns.get_sample_db(i_tsize=(256, 256),i_num_train_samples=1000,i_num_val_samples=100)
    train_images, train_masks, train_labels = ex_train_db
    val_images, val_masks, val_labels = ex_val_db
    trainer.train(i_train_db=list(zip(train_images, train_masks)), i_val_db=list(zip(val_images, val_masks)))
    trainer.eval(i_db=list(zip(train_images, train_masks)))
    trainer.eval(i_db=list(zip(val_images, val_masks)))
"""=================================================================================================================="""