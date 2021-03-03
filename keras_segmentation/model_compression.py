import keras
import tensorflow as tf

from tqdm import tqdm 
import numpy as np
import six 
import os 
import json 

from .data_utils.data_loader import image_segmentation_generator
from .predict import model_from_checkpoint_path
from .models.unet import unet_mini
from .train import CheckpointsCallback 



class Distiller(keras.Model):
    def __init__(self, student, teacher , distilation_loss ):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.distilation_loss = distilation_loss

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

        with tf.GradientTape() as tape:
            student_predictions = self.student( student_input , training=True)
            student_predictions_resize = tf.reshape(student_predictions , ((-1, self.student.output_height , self.student.output_width , self.student.output_shape[-1])))
            student_predictions_resize = tf.image.resize( student_predictions_resize , ( self.teacher.output_height , self.teacher.output_width ) )
            
            loss = self.distilation_loss( teacher_predictions_reshape , student_predictions_resize )

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
                        batch_size = 6 ,checkpoints_path=None  , epochs = 32 , steps_per_epoch=512, ):

    if isinstance( distilation_loss , six.string_types):
        if distilation_loss == "l1":
            distilation_loss = keras.losses.MeanAbsoluteError()
        elif distilation_loss == "l2":
            distilation_loss = keras.losses.MeanSquaredError()
        elif distilation_loss=="kl":
            # prolly we have to make it 1d first lol 
            distilation_loss = keras.losses.KLDivergence()
    
    distill_model = Distiller( student=student_model , teacher=teacher_model , distilation_loss=distilation_loss  )

    img_gen  = image_segmentation_generator(images_path=data_path , segs_path=None, batch_size=batch_size,
                                     n_classes=teacher_model.n_classes , input_height=teacher_model.input_height, input_width=teacher_model.input_width,
                                     output_height=None, output_width=None , ignore_segs=True)

    distill_model.compile(
        optimizer=keras.optimizers.Adam(),
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
    
    