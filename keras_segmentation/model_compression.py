import keras
import tensorflow as tf

from tqdm import tqdm 
import numpy as np
import six 
import os 
import json 
import sys 

from .data_utils.data_loader import image_segmentation_generator
from .train import CheckpointsCallback 

from keras.models import Model 



def get_pariwise_similarities( feats ):
    feats_i = tf.reshape( feats , (-1 , 1 , feats.shape[1]*feats.shape[2] , feats.shape[3]))
    feats_j = tf.reshape( feats , (-1 ,  feats.shape[1]*feats.shape[2] , 1 , feats.shape[3]))
    
    feats_i = feats_i / (( tf.reduce_sum(feats_i**2 , axis=-1 ) )**(0.5))[ ... , None ]
    feats_j = feats_j / (( tf.reduce_sum(feats_j**2 , axis=-1 ) )**(0.5))[ ... , None ]
    
    feats_ixj = feats_i*feats_j
    
    return tf.reduce_sum( feats_ixj , axis=-1  )
    

    
def pairwise_dist_loss( feats_t , feats_s ):
    
    # todo max POOL     
    pool_factor = 4
    
    feats_t = tf.nn.max_pool(feats_t , (pool_factor,pool_factor) , strides=(pool_factor,pool_factor) , padding="VALID" )
    feats_s = tf.nn.max_pool(feats_s , (pool_factor,pool_factor) , strides=(pool_factor,pool_factor) , padding="VALID" )
            
    sims_t  = get_pariwise_similarities( feats_t )
    sims_s = get_pariwise_similarities( feats_s )
    n_pixs = sims_s.shape[1]
    
    return tf.reduce_sum(tf.reduce_sum(((sims_t - sims_s )**2 ) , axis=1), axis=1)/(n_pixs**2 )



class Distiller(keras.Model):
    def __init__(self, student, teacher , distilation_loss , feats_distilation_loss=None , feats_distilation_loss_w=0.1   ):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.distilation_loss = distilation_loss
        
        self.feats_distilation_loss = feats_distilation_loss 
        self.feats_distilation_loss_w = feats_distilation_loss_w 
        
        if not feats_distilation_loss is None:
            try:
                s_feat_out = student.get_layer("seg_feats").output
            except:
                s_feat_out = student.get_layer(student.seg_feats_layer_name ).output
            
            
            try:
                t_feat_out = teacher.get_layer("seg_feats").output 
            except:
                t_feat_out = teacher.get_layer(teacher.seg_feats_layer_name ).output
            
            
            self.student_feat_model = Model( student.input , s_feat_out  )
            self.teacher_feat_model = Model( teacher.input , t_feat_out  )

    def compile(
        self,
        optimizer,
        metrics,

    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)


    def train_step(self, data):
        teacher_input ,  = data 
        
        student_input = tf.image.resize( teacher_input , ( self.student.input_height , self.student.input_width ) )

        teacher_predictions = self.teacher(teacher_input, training=False)
        teacher_predictions_reshape = tf.reshape(teacher_predictions , ((-1   , self.teacher.output_height , self.teacher.output_width , self.teacher.output_shape[-1])))
        
        
        if not self.feats_distilation_loss is None:
            teacher_feats = self.teacher_feat_model(teacher_input, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student( student_input , training=True)
            student_predictions_resize = tf.reshape(student_predictions , ((-1, self.student.output_height , self.student.output_width , self.student.output_shape[-1])))
            student_predictions_resize = tf.image.resize( student_predictions_resize , ( self.teacher.output_height , self.teacher.output_width ) )
            
            loss = self.distilation_loss( teacher_predictions_reshape , student_predictions_resize )
            
            if not self.feats_distilation_loss is None:
                student_feats = self.student_feat_model( student_input , training=True)
                student_feats_resize = tf.image.resize( student_feats , ( teacher_feats.shape[1] , teacher_feats.shape[2] ) )
                loss += self.feats_distilation_loss_w*self.feats_distilation_loss( teacher_feats , student_feats_resize )
            
            
            

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(teacher_predictions_reshape , student_predictions_resize )
        

        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {  "distillation_loss": loss}
        )
        return results


# created a simple custom fit generator due to some issue in keras
def fit_generator_custom( model , gen , epochs , steps_per_epoch , callback=None ):
    for ep in range( epochs ):
        print("Epoch %d/%d"%(ep+1 , epochs ))
        bar = tqdm( range(steps_per_epoch))
        losses = [ ]
        for i in bar:
            x = next( gen )
            l = model.train_on_batch( x  )  
            losses.append( l )
            bar.set_description("Loss : %s"%str(np.mean( np.array(losses)   )))
        if not callback is None:
            callback.model = model.student 
            callback.on_epoch_end( ep )
            
            
def perform_distilation(teacher_model ,student_model, data_path , distilation_loss='kl' , 
                        batch_size = 6 ,checkpoints_path=None  , epochs = 32 , steps_per_epoch=512,
                        feats_distilation_loss=None , feats_distilation_loss_w=0.1  ):
    
    
    losses_dict = { 'l1':keras.losses.MeanAbsoluteError() , "l2": keras.losses.MeanSquaredError() , "kl":keras.losses.KLDivergence() , 'pa':pairwise_dist_loss }

    if isinstance( distilation_loss , six.string_types):
        distilation_loss = losses_dict[ distilation_loss ]
        
    if isinstance( feats_distilation_loss , six.string_types):
        feats_distilation_loss = losses_dict[ feats_distilation_loss ]
        
    
    distill_model = Distiller( student=student_model , teacher=teacher_model , distilation_loss=distilation_loss, feats_distilation_loss=feats_distilation_loss  , feats_distilation_loss_w=feats_distilation_loss_w )

    img_gen  = image_segmentation_generator(images_path=data_path , segs_path=None, batch_size=batch_size,
                                     n_classes=teacher_model.n_classes , input_height=teacher_model.input_height, input_width=teacher_model.input_width,
                                     output_height=None, output_width=None , ignore_segs=True)

    distill_model.compile(
        optimizer='adam',
        metrics=[ distilation_loss  ]
    )
    
    if checkpoints_path is not None:
        config_file = checkpoints_path + "_config.json"
        dir_name = os.path.dirname(config_file)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(config_file, "w") as f:
            json.dump({
                "model_class": student_model.model_name,
                "n_classes": student_model.n_classes,
                "input_height": student_model.input_height,
                "input_width": student_model.input_width,
                "output_height": student_model.output_height,
                "output_width": student_model.output_width
            }, f)

        cb = CheckpointsCallback( checkpoints_path )
    else:
        cb = None 

    fit_generator_custom( distill_model , img_gen ,  steps_per_epoch=steps_per_epoch ,  epochs=epochs ,callback=cb )

    print("done ")
    
    
