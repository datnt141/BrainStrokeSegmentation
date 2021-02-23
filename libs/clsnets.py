import os
import imageio
import numpy as np
import tensorflow as tf
from libs.logs import Logs
import matplotlib.pyplot as plt
from libs.commons import SupFns
from scipy.special import softmax
from libs.commons import ListTuples
from libs.sysParams import SysParams
from libs.datasets.tfrecords import TFRecordDB
from libs.callbacks import CustomCallback,LearningRateScheduler
np_int_types = (np.int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
tf_int_types = (tf.dtypes.int8,tf.dtypes.int16,tf.dtypes.int32,tf.dtypes.int64,tf.dtypes.uint8,tf.dtypes.uint16,tf.dtypes.uint32,tf.dtypes.uint64)
"""=================================================================================================================="""
class ImageClsNets:
    vckpts         = None         # Checkpoint for storing data
    vmodel_name    = 'VGG16'      # Model name
    vinput_shape   = (224,224,3)  # Input image shape
    vnum_classes   = 2            # Number of target classes
    vtime_steps    = 1            # For time-sequence classification
    vlr            = 0.0001       # Initial Learning rate
    vloss          = 'swCE'       # Name of loss method
    vweights       = (0.45, 0.55) # For using weighted cross entropy
    vnum_epochs    = 100          # Number of training epochs
    vbatch_size    = 32           # Size of batch
    vdb_repeat     = 1            # Repeat dataset at single learing rate
    vtfrecord_size = 1000         # Size of tfrecord file
    vlsm_factor    = 0.0          # Label smoothing factor
    vcontinue      = True         # Continue training or not
    vflip_ud       = True         # Flip up-down in data augmentation
    vflip_lr       = True         # Flip left-right in data augmentation
    vcrop_ratio    = 0.1          # Crop ratio
    vdebug         = True         # Debuging flag
    def __init__(self):
        """1.Process check point"""
        if self.vckpts is None:
            self.vckpts = os.path.join(os.getcwd(),'ckpts')
        else:
            assert isinstance(self.vckpts,str), 'Got type: {}'.format(type(self.vckpts))
        if not os.path.exists(self.vckpts):
            os.makedirs(self.vckpts)
        else:
            pass
        """2. Process the model name"""
        if self.vmodel_name is None:
            self.vmodel_name = 'VGG16'#Default
        else:
            assert isinstance(self.vmodel_name,str), 'Got type: {}'.format(self.vmodel_name)
            assert self.vmodel_name in ClsNets.model_lists
        self.model_path = os.path.join(self.vckpts,'cls_{}.h5'.format(self.vmodel_name))
        crop_height     = int(self.vinput_shape[0]*(1-self.vcrop_ratio))
        crop_width      = int(self.vinput_shape[1]*(1-self.vcrop_ratio))
        self.crop_size  = (crop_height,crop_width,self.vinput_shape[2])
        """3.Assertion"""
        assert isinstance(self.vinput_shape,(list,tuple)), 'Got type: {}'.format(type(self.vinput_shape))
        assert isinstance(self.vnum_classes,int),'Got type: {}'.format(type(self.vnum_classes))
        assert self.vnum_classes>1, 'Got value: {}'.format(self.vnum_classes)
        assert isinstance(self.vtime_steps,int),'Got type: {}'.format(type(self.vtime_steps))
        assert self.vtime_steps>0,'Got value: {}'.format(self.vtime_steps)
        assert isinstance(self.vloss,str),'Got type: {}'.format(type(self.vloss))
        assert self.vloss in ('wCE','swCE','FL','CE'), 'Got value: {}'.format(self.vloss)
        assert isinstance(self.vcontinue,bool),'Got type: {}'.format(type(self.vcontinue))
        """Load pretrained model if existed"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path,custom_objects=ClsNets.get_custom_objects())
        else:
            self.model = None
    def init_params(self,i_params):
        assert isinstance(i_params, (SysParams,dict)), 'Got type: {}'.format(type(i_params))
        Logs.log('-'*100)
        Logs.log('Init clsnet class global variables...')
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
                assert isinstance(key, str), 'Got type: {}'.format(type(key))
                if key.startswith('v'):
                    self.__dict__.update({key: val})
                    Logs.log('Init {} as : {}'.format(key, val))
                else:
                    pass
        """Custom parameter adjustment"""
        self.__init__()
        Logs.log('Set-up parameters for classification nets:')
        Logs.log_cls_params(self)
        return True
    """Training function"""
    def train(self,i_train_db=None,i_val_db=None):
        """i_train_db is a tfrecord dataset that contain (image,label) pair"""
        """i_val_db is similar to i_train_db (or None) that is used for evaluation of network"""
        if i_train_db is None:#Only valid if we alredy save data to the tfrecord dataset before.
            i_train_db = self.load_data(i_db=i_train_db,i_train_flag=True)
            assert isinstance(i_train_db,tf.data.Dataset)
        else:
            pass
        assert isinstance(i_train_db, (list,tuple,tf.data.Dataset)), 'Got type: {}'.format(type(i_train_db))
        if isinstance(i_train_db,(list,tuple)):
            train_db = self.load_data(i_db=i_train_db,i_train_flag=True)
            if i_val_db is None:
                val_db = None
            else:
                if isinstance(i_val_db,(list,tuple)):
                    val_db = self.load_data(i_val_db,i_train_flag=False)
                else:
                    assert isinstance(i_val_db,tf.data.Dataset),'Got type: {}'.format(type(i_val_db))
                    val_db = i_val_db
            return self.train(i_train_db=train_db,i_val_db=val_db)
        else:
            assert isinstance(i_train_db,tf.data.Dataset), 'Got type: {}'.format(type(i_train_db))
            train_db = i_train_db.batch(self.vbatch_size)
            if i_val_db is None:
                val_db = None
            else:
                if isinstance(i_val_db,(list,tuple)):
                    val_db = self.load_data(i_val_db,i_train_flag=False)
                    val_db = val_db.batch(self.vbatch_size)
                else:
                    assert isinstance(i_val_db,tf.data.Dataset), 'Got type: {}'.format(type(i_val_db))
                    val_db = i_val_db.batch(self.vbatch_size)
        """Start training using tfrecord dataset"""
        if os.path.exists(self.model_path):
            if self.vcontinue:
                Logs.log('Continue training')
                net = tf.keras.models.load_model(self.model_path,custom_objects=ClsNets.get_custom_objects())
            else:
                return False
        else:
            Logs.log('Train model from scratch!')
            net = ClsNets(i_model_name  = self.vmodel_name,
                          i_image_shape = self.vinput_shape,
                          i_time_steps  = self.vtime_steps,
                          i_num_classes = self.vnum_classes).build()
        net.summary(print_fn=Logs.log)
        net = ClsNets.compile(i_net=net, i_lr=self.vlr, i_loss_name=self.vloss,i_weights=self.vweights)
        """Debugging"""
        if self.vdebug:
            cnt      = 0
            num_test = 5
            for batch in train_db:
                dimages,dlabels = batch
                for dindex,dimage in enumerate(dimages):
                    dlabel = dlabels[dindex]
                    plt.imshow(dimage,cmap='gray')
                    plt.title('Label = {}'.format(np.argmax(dlabel.numpy())))
                    plt.show()
                    cnt +=1
                    if cnt>num_test:
                        break
                    else:
                        pass
                    print('debug_count = ',cnt)
                if cnt>num_test:
                    break
                else:
                    pass
        else:
            pass
        """Training"""
        log_infor  = CustomCallback(i_model_path=self.model_path)
        lr_schuler = LearningRateScheduler()
        lr_params  = {'decay_rule': 1, 'step': int(self.vnum_epochs / 5), 'decay_rate': 0.90, 'base_lr': self.vlr}
        schedule   = lr_schuler(lr_params)
        callbacks  = [schedule, log_infor]
        net.fit(x               = train_db.repeat(self.vdb_repeat),
                epochs          = self.vnum_epochs,
                verbose         = 1,
                shuffle         = True,
                validation_data = val_db,
                callbacks       = callbacks)
        """Update the nework"""
        self.model = tf.keras.models.load_model(self.model_path, custom_objects=ClsNets.get_custom_objects())
        return net
    """Performance evaluation of the trained network using test data"""
    def eval(self,i_db=None):
        """i_db is single-batch image (each element contains only one image and its correponding label)"""
        """i_db can be list (tuple) of (image,label) or tfrecord dataset"""
        assert isinstance(i_db,(list,tuple,tf.data.Dataset)), 'Got type: {}'.format(type(i_db))
        labels, preds = [],[]
        for index, element in enumerate(i_db):#Process for every single image
            print('(ClsNet) Evaluating index = {}'.format(index))
            assert isinstance(element,(list,tuple,dict))
            if isinstance(element,(list,tuple)):
                image = element[0] #As my design. Shape = (height, width, nchannels)
                label = element[1] #As my design. Shape = (num_classes,) (one-hot)
            else:
                image = element['image'] #As my design. Shape = (height, width, nchannels)
                label = element['label'] #As my design. Shape = (num_classes,) (one-hot)
            if isinstance(image,(tf.Tensor,tf.SparseTensor)):
                image = image.numpy()
            else:
                assert isinstance(image,np.ndarray)
            if isinstance(label,(tf.Tensor,tf.SparseTensor)):
                label = label.numpy()
            else:
                label = label.astype(np.int)
            assert isinstance(image,np.ndarray), 'Got type: {}'.format(type(image))
            """Prepare data"""
            assert len(image.shape) in (2, 3, 4)
            if len(image.shape) in (2, 3):  # Single image
                if len(image.shape) == 2:  # Gray image with shape (height,width)
                    image = np.expand_dims(image, axis=-1)  # Shape: (height, width, 1)
                else:  # Shape: (height, width, depth)
                    assert image.shape[-1] in (1, 3)
                """Making batch for single image"""
                assert label.dtype in np_int_types, 'Got type: {}'.format(type(label))
                image = np.expand_dims(image, 0) #Shape: (1, height, width, depth)
                label = np.expand_dims(label, 0) #Shape: (1, )
            else:  # Batch of images
                assert isinstance(label,np.ndarray),'Got type: {}'.format(type(label))
                assert len(label.shape) == 1, 'Got shape: {}'.format(label.shape)  # Shape: (None, )
                assert len(image.shape) == 4, 'Got shape: {}'.format(image.shape)  # Shape: (None, height, width, depth)
                assert len(label.shape) == image.shape[0], 'Got shape: {} vs {}'.format(label.shape,image.shape) #Same batch size
            """Prediction"""
            cpreds = self.predict(i_image=image) #Shape (None,num_classes)
            for pindex, cpred in enumerate(cpreds):
                labels.append(label[pindex])      # As my design. Shape = (num_classes,) (one-hot)
                preds.append(cpred)               # As my design of predict(): Shape = (num_classes, ) of probability
        """Performance measurement"""
        matrix = np.zeros(shape=(self.vnum_classes,self.vnum_classes))
        for index, pred in enumerate(preds):
            label = labels[index]
            if isinstance(label,np.ndarray):
                label = np.argmax(label)
            else:
                label = int(label)
            if isinstance(pred,np.ndarray):
                pred = np.argmax(pred)
            else:
                pred = int(pred)
            matrix[label][pred]+=1
        Logs.log_matrix(i_str='Confusion Matrix',i_matrix=matrix)
        sum_matrix = np.sum(matrix)
        corrects   = np.sum(np.diag(matrix))
        Logs.log('Accuracy = {}/{} ~ {} (%)'.format(corrects,sum_matrix,100*corrects/sum_matrix))
        return preds
    """Main function for prediction"""
    def predict(self,i_image=None):
        assert isinstance(self.model,tf.keras.models.Model),'Got type: {}'.format(type(self.model))
        assert isinstance(i_image,np.ndarray),'Got type: {}'.format(type(i_image))
        assert len(i_image.shape) in (2, 3,4) #Single image or batch of images
        if len(i_image.shape) in (2,3):#Single image with shape (height, width) or (height, width, depth)
            if len(i_image.shape)==2: #Single image with shape (height, width)
                images = np.expand_dims(i_image,-1)
            else:
                assert len(i_image.shape)==3
                assert i_image.shape[-1] in (1, 3)
                images = i_image.copy()
            """Making batch of image with num_batch = 1"""
            images = np.expand_dims(images, 0)
        else:#Batch Of Images
            assert i_image.shape[-1] in (1,3)
            images = i_image.copy()
        assert len(images.shape) == 4, 'Got shape: {}'.format(images.shape)
        assert images.dtype in (np.uint8, ), 'Got dtype: {}'.format(images.dtype)
        """Size Normalization"""
        nimages = []
        for image in images:
            assert isinstance(image, np.ndarray), 'Got type: {}'.format(type(image))
            """Color adjustment"""
            if image.shape[-1] == self.vinput_shape[-1]:
                pass
            else:
                if image.shape[-1] == 1:  # Convert to color image
                    image = np.concatenate((image, image, image), axis=-1)
                else:  # Convert to gray image
                    image = (np.average(image, axis=-1)).astype(np.uint8)
            """Resizing"""
            nimages.append(SupFns.imresize(i_image=image,i_tsize=self.vinput_shape[0:2]))
        images = np.array(nimages) #Always in uint8 format as they are outputs of SupFuns.imresize()
        assert len(images.shape) == 4 , 'Got shape: {} with length = {}'.format(images.shape,len(images.shape))
        assert images.shape[-1] in (1, 3), 'Got shape: {}'.format(images.shape)
        """Normalization"""
        images = images/255.0
        """Prediction"""
        preds  = self.model.predict(images) #Shape  = (None, num_classes)
        return softmax(preds,axis=-1)       #Shape  = (None, num_classes). As my design, output of network is logit. So we do softmax here
    """Data preparation"""
    def pipeline(self,i_record=None,i_ori_shape=None,i_train_flag=True):
        """i_record is (image, label) where image is uint8 normal image, label is integer label"""
        """i_ori_shape --- Original shape of input image"""
        assert isinstance(i_record,(list,tuple,dict)), 'Got type: {}'.format(type(i_record))
        assert isinstance(i_ori_shape, (list, tuple)),'Got type: {}'.format(i_ori_shape)
        assert isinstance(i_train_flag,bool), 'Got type: {}'.format(type(i_train_flag))
        assert len(i_ori_shape) in (2,3)
        if len(i_ori_shape)==2:
            i_ori_shape = (i_ori_shape[0],i_ori_shape[1],1)
        else:
            assert i_ori_shape[-1] in (1, 3), 'Got value: {}'.format(i_ori_shape)
        if isinstance(i_record,(list,tuple)):
            image = i_record[0]  #uint8 image
            label = i_record[1]  #int label
        else:
            assert isinstance(i_record,dict)
            image = i_record['image']#uint8 image
            label = i_record['label']#int label
        """Processing image"""
        assert isinstance(image, (tf.Tensor, tf.SparseTensor))
        assert isinstance(label,(tf.Tensor,tf.SparseTensor))
        assert image.dtype in (tf.dtypes.uint8,)
        assert label.dtype in tf_int_types
        image = tf.reshape(tensor=image, shape=i_ori_shape)
        if i_ori_shape[-1]==self.vinput_shape[-1]:
            pass
        else:
            if i_ori_shape[-1] == 1:
                image = tf.concat(values=(image,image,image),axis=-1)
            else:
                image = tf.reshape(tf.cast(tf.reduce_mean(input_tensor=image,axis=-1),tf.dtypes.uint8),shape=(i_ori_shape[0],i_ori_shape[1],1))
        image = tf.cast(tf.image.resize(image, size=(self.vinput_shape[0], self.vinput_shape[1])),tf.dtypes.uint8)
        label = tf.cast(label,tf.dtypes.int32)
        if i_train_flag:  # Training dataset
            if self.vflip_ud:
                image = tf.image.random_flip_up_down(image)
            else:
                pass
            if self.vflip_lr:
                image = tf.image.random_flip_left_right(image)
            else:
                pass
            image = tf.image.random_crop(value=image, size=self.crop_size)
            image = tf.image.resize(image, size=self.vinput_shape[0:2])
            """Normalization"""
            image = tf.cast(image, tf.float32) / 255.0
            """Label smoothing"""
            label = tf.cast(tf.one_hot(label, self.vnum_classes), tf.dtypes.float32)
            label = label * (1.0 - self.vlsm_factor)
            label = label + self.vlsm_factor / self.vnum_classes
        else:
            """Normalization"""
            image = tf.reshape(image,shape=self.vinput_shape)
            image = tf.cast(image, tf.float32) / 255.0
            label = tf.cast(tf.one_hot(label, self.vnum_classes), tf.float32)
        return image, label
    def load_data(self,i_db=None,i_train_flag=True):
        """This function is to prepare data for training or testing using conventional list of images"""
        """As my design, i_db is list of (image,label) or (image_path,label)"""
        """Note: i_db can be None if it already saved as tfrecord file at the specified checkpoint directory"""
        assert isinstance(i_train_flag,bool), 'Got type: {}'.format(type(i_train_flag))
        tf_save_path = os.path.join(self.vckpts, 'tfrecords')
        if not os.path.exists(tf_save_path):
            os.makedirs(tf_save_path)
        else:
            pass
        if i_train_flag:
            i_save_path = os.path.join(tf_save_path,'cls_train_db.tfrecord')
        else:
            i_save_path = os.path.join(tf_save_path,'cls_val_db.tfrecord')
        if os.path.exists(i_save_path):
            dataset = TFRecordDB.read(i_tfrecord_path=i_save_path,i_original=True) #Set i_original to True to return dictionary
        else:
            assert isinstance(i_db, (list, tuple))
            dataset  = self.prepare_db(i_db=i_db,i_save_path=i_save_path)
        dataset = dataset.map(lambda x:self.pipeline(i_record=x,i_ori_shape=self.vinput_shape,i_train_flag=i_train_flag))
        return dataset
    """Prepare dataset as tf.data.Dataset object. DON'T USE THIS FUNCTION DIRECTLY. USE load_data() instead"""
    def prepare_db(self,i_db=None,i_save_path=None):
        """i_db is a list (tuple) of (image,label) or (image_path,label)"""
        assert isinstance(i_db, (list, tuple)), 'Got type: {}'.format(type(i_db))
        assert isinstance(i_save_path,str),'Got type: {}'.format(type(i_save_path))
        assert i_save_path.endswith('.tfrecord')
        save_path,save_file = os.path.split(i_save_path)
        save_file     = save_file[0:len(save_file)-len('.tfrecord')]
        db_fields     = {'image': [], 'label': []}
        images,labels = [],[]
        count         = 0
        segment       = 0
        TFRecordDB.lossy = False
        tfwriter         = TFRecordDB()
        for index,element in enumerate(i_db):
            image,label = element # As my design
            if isinstance(image,str):#i_db is (image_path,label) pair
                assert os.path.exists(image)
                image = imageio.imread(image)
            else:#i_db is (image,label) pair
                pass
            if type(label) in np_int_types:
                label = int(label)
            else:
                assert isinstance(label,int)
            assert label>=0, 'Got value: {}'.format(label)
            """Image processing"""
            assert len(image.shape) in (2,3), 'Got shape: {}'.format(image.shape)
            if len(image.shape)==2:
                image = np.expand_dims(image,-1)
            else:
                assert image.shape[-1] in (1, 3), 'Got shape: {}'.format(image.shape)
            """Color adjustment"""
            if image.shape[-1] == self.vinput_shape[-1]:
                pass
            else:
                if image.shape[-1]==1:
                    image = np.concatenate((image,image,image),axis=-1)
                else:
                    image = (np.average(image,axis=-1)).astype(np.uint8)
            """Size adjustment"""
            if np.sum(ListTuples.compare(i_x=image.shape,i_y=self.vinput_shape)):
                image = SupFns.imresize(i_image=image,i_tsize=self.vinput_shape[0:2])
            else:
                pass
            assert isinstance(image,np.ndarray),'Got type: {}'.format(type(image))
            assert image.dtype in (np.uint8,), 'Got dtype: {}'.format(image.dtype)
            images.append(image)
            labels.append(label)
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
        if len(images)>0:
            if segment==0:
                csave_path = os.path.join(save_path,'{}.tfrecord'.format(save_file))
            else:
                csave_path = os.path.join(save_path, '{}_{}.tfrecord'.format(save_file, segment))
            write_data = list(zip(images, labels))
            tfwriter.write(i_n_records=write_data, i_size=self.vtfrecord_size, i_fields=db_fields,i_save_file=csave_path)
        else:
            pass
        dataset = tfwriter.read(i_tfrecord_path=i_save_path,i_original=True)
        return dataset
"""=================================================================================================================="""
"""=================================================================================================================="""
class ClsNets:
    """Please update this one if we newly updated a new model"""
    model_lists = ('VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'Inception', 'InceptionResNet', 'DenseNet121',
                   'DenseNet169', 'DenseNet201', 'cCNN_MS', 'cCNN_MS_tiny', 'cCNN', 'cCNN_tiny', 'cCNN_tiny_tiny','attCNN')
    def __init__(self,
                i_model_name  = 'VGG16',
                i_time_steps  = 2,
                i_image_shape = (256,256,3),
                i_fine_tune   = True,
                i_num_classes = 2):
        assert isinstance(i_time_steps, int)
        assert i_time_steps > 0
        self.model_name  = i_model_name
        self.image_shape = i_image_shape
        self.time_step   = i_time_steps
        self.fine_tune   = i_fine_tune
        self.num_classes = i_num_classes
        """Calculate the input shape"""
        if self.time_step == 1:
            self.input_shape = self.image_shape
        else:
            self.input_shape = (self.time_step,) + self.image_shape
        assert self.model_name in self.model_lists
    def build(self):
        p_shape = (self.image_shape[-3],self.image_shape[-2],3) #Input shape of popular nets
        if self.model_name == 'VGG16':
            model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == 'VGG19':
            model = tf.keras.applications.VGG19(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == 'ResNet50':
            model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == 'ResNet101':
            model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == 'ResNet152':
            model = tf.keras.applications.ResNet152(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == 'Inception':
            model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == 'InceptionResNet':
            model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == "DenseNet121":
            model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == "DenseNet169":
            model = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == "DenseNet201":
            model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet',input_shape=p_shape)
        elif self.model_name == 'cCNN_MS':#Custom CNN Multiscale
            return ClsNets.custom_cnn(i_input_shape=self.image_shape,i_filters=32,i_multiscale=True,i_num_classes=self.num_classes)
        elif self.model_name == 'cCNN_MS_tiny':#Custom CNN Multiscale
            return ClsNets.custom_cnn_tiny(i_input_shape=self.image_shape,i_filters=32,i_multiscale=True,i_num_classes=self.num_classes)
        elif self.model_name == 'cCNN':#Custom CNN
            return ClsNets.custom_cnn(i_input_shape=self.image_shape, i_filters=32, i_multiscale=False,i_num_classes=self.num_classes)
        elif self.model_name == 'cCNN_tiny':#Custom CNN
            return ClsNets.custom_cnn_tiny(i_input_shape=self.image_shape, i_filters=32, i_multiscale=False,i_num_classes=self.num_classes)
        elif self.model_name == 'cCNN_tiny_tiny':
            return ClsNets.custom_cnn_tiny_tiny(i_input_shape=self.image_shape, i_filters=32, i_multiscale=False,i_num_classes=self.num_classes)
        elif self.model_name == 'attCNN':
            assert self.input_shape[0]>=64
            assert self.input_shape[1]>=64
            num_stages = self.input_shape[0]//64
            return ClsNets.attention_nets(i_input_shape=self.image_shape,i_filters=32,i_num_classes=self.num_classes,i_num_stages=num_stages)
        else:#Default is VGG16
            model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',input_shape=p_shape)
        model.trainable = False if self.fine_tune else True
        """1. Output of network"""
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        if self.input_shape[-1]!=3:
            outputs = tf.keras.layers.Conv2D(filters=3,kernel_size=(3,3),strides=(1,1),padding='same')(inputs)
        else:
            outputs=inputs
        if self.time_step==1:#Conventional CNN
            outputs = model(outputs)
            """2. Futher manipulation"""
            if self.fine_tune:
                outputs = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),activation='relu')(outputs)
            else:
                pass
            outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.GaussianNoise(stddev=0.1)(outputs)
        else:#Sequences
            outputs = tf.keras.layers.TimeDistributed(model)(outputs)
            """2. Futher manipulation"""
            if self.fine_tune:
                outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu'))(outputs)
            else:
                pass
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(outputs)
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(outputs)
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.GaussianNoise(stddev=0.1))(outputs)
            """Combine feature"""
            outputs = tf.keras.layers.Flatten()(outputs)
        """Classification layers"""
        outputs = tf.keras.layers.Dense(units=self.num_classes, activation=None, name='combine')(outputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model
    @staticmethod
    def residual_block(i_inputs, i_kernel_size, i_nb_filters, i_stride=(1, 1), use_bias=True):
        """Custom network using residual connection blocks"""
        assert isinstance(i_kernel_size, int)
        assert isinstance(i_nb_filters, (list, tuple))
        assert len(i_nb_filters) == 3
        assert i_kernel_size>=3
        nb_filter1, nb_filter2, nb_filter3 = i_nb_filters
        """First CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter1,kernel_size=(1, 1), strides=(1,1), use_bias=use_bias)(i_inputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Second CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter2,kernel_size=(i_kernel_size, i_kernel_size),strides=i_stride,padding='same', use_bias=use_bias)(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Third CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter3, kernel_size=(1, 1),strides=(1,1),use_bias=use_bias)(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        """Shortcut CONV block"""
        shortcut = tf.keras.layers.Conv2D(nb_filter3, (3, 3), strides=i_stride,padding='same', use_bias=use_bias)(i_inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        """Aggregation"""
        outputs = tf.keras.layers.Add()([outputs, shortcut])
        outputs = tf.keras.layers.Activation('relu')(outputs)
        return outputs
    @staticmethod
    def attention_block(i_inputs,i_att_inputs,i_nb_filters=32):
        """Perform attention block"""
        """Reference: Attention Enriched Deep Learning Model for Brest Tumor Segmentation in Ultrasound Images"""
        pool_outputs = tf.keras.layers.MaxPool2D(pool_size=(2,2))(i_inputs)
        outputs      = tf.keras.layers.Conv2D(filters=i_nb_filters,kernel_size=(3,3),strides=(2,2),padding='same')(i_inputs)
        outputs      = tf.keras.layers.Activation('relu')(outputs)
        att_outputs  = tf.keras.layers.Conv2D(filters=i_nb_filters,kernel_size=(3,3),strides=(1,1),padding='same')(i_att_inputs)
        att_outputs  = tf.keras.layers.Activation('relu')(att_outputs)
        outputs      = tf.keras.layers.Add()([outputs,att_outputs])
        outputs      = tf.keras.layers.Conv2D(filters=i_nb_filters,kernel_size=(3,3),strides=(1,1),padding='same')(outputs)
        outputs      = tf.keras.layers.Activation('relu')(outputs)
        outputs      = tf.keras.layers.Conv2D(filters=i_nb_filters, kernel_size=(3, 3), strides=(1, 1),padding='same')(outputs)
        outputs      = tf.keras.layers.Activation('relu')(outputs)
        outputs      = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),padding='same')(outputs)
        outputs      = tf.keras.layers.Activation('sigmoid')(outputs)
        outputs      = tf.keras.layers.Multiply()([pool_outputs,outputs])
        return outputs
    @staticmethod
    def attention_nets(i_input_shape = (256,256,3),i_filters=32,i_num_classes=2,i_num_stages=6):
        height     = i_input_shape[0]
        width      = i_input_shape[1]
        mask_shape = (height,width,1)
        inputs = tf.keras.layers.Input(shape=i_input_shape)
        imasks  = tf.keras.layers.Input(shape=mask_shape)
        """1. Warming-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu')(inputs)
        masks = imasks
        for stage in range(i_num_stages):
            filters = i_filters * (stage+1)
            height  = height//2
            width   = width//2
            outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
            outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
            masks   = tf.image.resize(images=masks, size=(height, width)) #Use image resize, not Pooling to make uncertainty of boundary
            outputs = ClsNets.attention_block(i_inputs=outputs,i_att_inputs=masks,i_nb_filters=i_filters)
            print('mask {} and outputs {}'.format(masks.shape,outputs.shape))
        outputs = tf.keras.layers.Flatten()(outputs)
        outputs = tf.keras.layers.Dropout(rate=0.250)(outputs)
        outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(outputs)
        return tf.keras.models.Model(inputs=[inputs,imasks], outputs=[outputs])
    """Multi-scale CNN"""
    @staticmethod
    def custom_cnn(i_input_shape, i_filters=32, i_multiscale=False,i_num_classes=2):
        assert isinstance(i_num_classes, int)
        assert i_num_classes > 0
        ms_outputs = []
        inputs = tf.keras.layers.Input(shape=i_input_shape)
        """1. Warming-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu')(inputs)
        """2. Residual connection layers"""
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(tf.keras.layers.MaxPool2D()(outputs))
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, 4 * filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        if i_multiscale:
            for index, outputs in enumerate(ms_outputs):
                ms_outputs[index] = tf.keras.layers.Flatten()(outputs)
            ms_outputs = tf.keras.layers.Concatenate(axis=-1)(ms_outputs)
            ms_outputs = tf.keras.layers.Dropout(rate=0.250)(ms_outputs)
            ms_outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(ms_outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[ms_outputs])
        else:
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dropout(rate=0.250)(outputs)
            outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    @staticmethod
    def custom_cnn_tiny(i_input_shape, i_filters=32, i_multiscale=False,i_num_classes=2):
        assert isinstance(i_num_classes, int)
        assert i_num_classes > 0
        ms_outputs = []
        inputs = tf.keras.layers.Input(shape=i_input_shape)
        """1. Warming-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu')(inputs)
        """2. Residual connection layers"""
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(tf.keras.layers.MaxPool2D()(outputs))
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, 4 * filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        if i_multiscale:
            for index, outputs in enumerate(ms_outputs):
                ms_outputs[index] = tf.keras.layers.Flatten()(outputs)
            ms_outputs = tf.keras.layers.Concatenate(axis=-1)(ms_outputs)
            ms_outputs = tf.keras.layers.Dropout(rate=0.250)(ms_outputs)
            ms_outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(ms_outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[ms_outputs])
        else:
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dropout(rate=0.250)(outputs)
            outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    @staticmethod
    def custom_cnn_tiny_tiny(i_input_shape, i_filters=32, i_multiscale=False,i_num_classes=2):
        assert isinstance(i_num_classes,int)
        assert i_num_classes>0
        ms_outputs = []
        inputs = tf.keras.layers.Input(shape=i_input_shape)
        """1. Warming-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu')(inputs)
        """2. Residual connection layers"""
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 6
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        filters = i_filters * 10
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, 4 * filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        if i_multiscale:
            for index, outputs in enumerate(ms_outputs):
                ms_outputs[index] = tf.keras.layers.Flatten()(outputs)
            ms_outputs = tf.keras.layers.Concatenate(axis=-1)(ms_outputs)
            ms_outputs = tf.keras.layers.Dropout(rate=0.250)(ms_outputs)
            ms_outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(ms_outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[ms_outputs])
        else:
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dropout(rate=0.250)(outputs)
            outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    @staticmethod
    def weighted_ce(i_weights):
        assert isinstance(i_weights, (list,tuple,tf.Tensor,tf.SparseTensor)) #List or Tuple of class weights
        epsilon = tf.keras.backend.epsilon()
        if isinstance(i_weights,(list,tuple)):
            weights = tf.convert_to_tensor(i_weights)  # Convert to tensor
        else:
            weights = tf.cast(i_weights,tf.float32)
        def get_wce(i_y_true,i_y_pred):
            """i_y_true is Tensor in shape [None, num_classes] in form of one-hot encoding"""
            """i_y_pred is Tensor in shape [None, num_classes] in form of unscaled (original) digits"""
            """tf.nn.sparse_softmax_cross_entropy_with_logits requires unscaled logit"""
            y_true  = tf.cast(i_y_true, tf.float32)                    # Result: (None, num_classes)
            y_pred  = tf.nn.softmax(i_y_pred, axis=-1)                 # Result: (None, num_classes)
            y_pred  = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Result: (None,  num_classes)
            wce     = -y_true * tf.math.log(y_pred) * weights          # Broast-casting. #Result: (None, num_classes)
            return tf.reduce_mean(tf.reduce_sum(wce, axis=-1), axis=None)
        return get_wce
    @staticmethod
    def sweighted_ce(i_y_true, i_y_pred):
        """i_y_true is Tensor in shape [None, num_classes] in form of one-hot encoding"""
        """i_y_pred is Tensor in shape [None, num_classes] in form of unscaled (original) digits"""
        """tf.nn.sparse_softmax_cross_entropy_with_logits requires unscaled logit"""
        y_true  = tf.cast(i_y_true, tf.float32)
        samples = tf.reduce_sum(y_true, axis=0)                  # Result: (num_classes,). Number of samples in each classes
        weights = tf.divide(samples, tf.reduce_sum(samples))       # Result: (num_classes,). Summation = 1.
        return ClsNets.weighted_ce(i_weights=weights)(i_y_true=i_y_true,i_y_pred=i_y_pred)
    @staticmethod
    def focal_loss(i_gamma=2.0, i_alpha=0.25):
        epsilon = tf.keras.backend.epsilon()
        def get_fl(i_y_true, i_y_pred):
            """i_y_true is Tensor in shape [None,num_classes] in form of one-hot encoding"""
            """i_y_pred is Tensor in shape [None, num_classes] in form of unscaled (original) digits"""
            y_true = tf.cast(i_y_true, tf.float32)     # Result: (None, num_classes)
            y_pred = tf.nn.softmax(i_y_pred, axis=-1)  # Result: (None,  num_classes)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Result: (None,  num_classes)
            ce     = -y_true * tf.math.log(y_pred)                    # Result: (None,  num_classes)
            loss = i_alpha * tf.math.pow(1.0 - y_pred, i_gamma) * ce  # Result: (None,  num_classes)
            return tf.reduce_mean(tf.reduce_sum(loss, axis=-1), axis=None)
        return get_fl
    @staticmethod
    def compile(i_net=None,i_lr=0.001,i_loss_name='wCE',i_weights=None):
        assert isinstance(i_loss_name,str)
        assert i_loss_name in('wCE','swCE','FL','CE')
        assert isinstance(i_weights,(list,tuple)) #Only used for 'wCE'
        if i_loss_name == 'wCE':
            loss  = ClsNets.weighted_ce(i_weights=i_weights)
        elif i_loss_name == 'swCE':
            loss  = ClsNets.sweighted_ce
        elif i_loss_name == 'FL':
            loss  = ClsNets.focal_loss(i_gamma=2.0,i_alpha=0.25)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        i_net.compile(optimizer  = tf.keras.optimizers.Adam(lr=i_lr),
                      loss       = loss,
                      metrics    = ['accuracy'])
        return i_net
    @staticmethod
    def get_custom_objects():
        custom_objects = {'get_wce'     : ClsNets.weighted_ce(i_weights=[0.45,0.55]),
                          'sweighted_ce': ClsNets.sweighted_ce,
                          'get_fl'      : ClsNets.focal_loss(i_gamma=2.,i_alpha=0.25)}
        return custom_objects
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module is to implement general classification networks.Check again!(2021-01-12)')
    print('Please set the vckpts carefully as all of code and results will be stored and look at this directory as my design')
    cls_params = SysParams()
    cls_params.vckpts         = None           # Checkpoint for storing data
    cls_params.vmodel_name    = 'VGG16'        # Model name
    cls_params.vinput_shape   = (128, 128, 1)  # Input image shape
    cls_params.vnum_classes   = 10             # Number of target classes
    cls_params.vtime_steps    = 1              # For time-sequence classification
    cls_params.vlr            = 0.0001         # Initial Learning rate
    cls_params.vloss          = 'swCE'         # Name of loss method
    cls_params.vweights       = (0.45, 0.55)   # For using weighted cross entropy
    cls_params.vnum_epochs    = 100            # Number of training epochs
    cls_params.vbatch_size    = 32             # Size of batch
    cls_params.vdb_repeat     = 1              # Repeat dataset at single learing rate
    cls_params.vtfrecord_size = 10000          # Size of tfrecord file
    cls_params.vcontinue      = False          # Continue training or not
    cls_params.vflip_ud       = False          # Flip up-down in data augmentation
    cls_params.vflip_lr       = False          # Flip left-right in data augmenation
    cls_params.vdebug         = True           # Debug flag
    trainer = ImageClsNets()
    trainer.init_params(i_params=cls_params)
    """Get sample dataset"""
    ex_train_db,ex_val_db = SupFns.get_sample_db(i_tsize=(256,256),i_num_train_samples=1000,i_num_val_samples=100)
    train_images, train_masks, train_labels = ex_train_db
    val_images, val_masks, val_labels       = ex_val_db
    trainer.train(i_train_db=list(zip(train_images,train_labels)),i_val_db=list(zip(val_images,val_labels)))
    trainer.eval(i_db=list(zip(train_images,train_labels)))
    trainer.eval(i_db=list(zip(val_images,val_labels)))
"""=================================================================================================================="""