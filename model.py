""" --------------------------------------------------
    author: arthur meyer
    email: arthur.meyer.38@gmail.com  
    status: final
    version: v2.0
    --------------------------------------------------"""



from __future__ import division
import tensorflow as tf
import numpy as np



class MODEL(object):
  """ 
  Model description:
    conv          :   vgg
    deconv        :   vgg + 1 more
    fc layer      :   2
    loss          :   flexible
    direct 
     connections  :   flexible (if yes then 111 110)
    edge contrast :   flexible
  """
  
  def __init__(self, name, batch_size, learning_rate, wd, concat, l2_loss, penalty, coef):
    """
    Args:
       name             : name of the model (used to create a specific folder to save/load parameters)
       batch_size       : batch size
       learning_rate    : learning_rate
       wd               : weight decay factor
       concat           : does this model include direct connections?
       l2_loss          : does this model use l2 loss (if not then cross entropy)
       penalty          : whether to use the edge contrast penalty
       coef             : coef for the edge contrast penalty
    """

    self.name                   =        'saliency_' + name
    self.losses                 =        'loss_of_' + self.name
    self.losses_decay           =        'loss_of_' + self.name +'_decay'
    self.batch_size             =        batch_size
    self.learning_rate          =        learning_rate
    self.wd                     =        wd
    self.moving_avg_decay       =        0.9999
    self.concat                 =        concat
    self.l2_loss                =        l2_loss
    self.penalty                =        penalty
    self.coef                   =        coef
    self.parameters_conv        =        []    
    self.parameters_deconv      =        []  
    self.deconv                 =        []  
    
    with tf.device('/cpu:0'):
      # conv1_1
      with tf.variable_scope(self.name + '_' + 'conv1_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 3, 64), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv1_2
      with tf.variable_scope(self.name + '_' + 'conv1_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 64, 64), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv2_1
      with tf.variable_scope(self.name + '_' + 'conv2_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 64, 128), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      # conv2_2
      with tf.variable_scope(self.name + '_' + 'conv2_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 128, 128), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv3_1
      with tf.variable_scope(self.name + '_' + 'conv3_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 128, 256), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv3_2
      with tf.variable_scope(self.name + '_' + 'conv3_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 256, 256), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      # conv3_3
      with tf.variable_scope(self.name + '_' + 'conv3_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 256, 256), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv4_1
      with tf.variable_scope(self.name + '_' + 'conv4_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 256, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv4_2
      with tf.variable_scope(self.name + '_' + 'conv4_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv4_3
      with tf.variable_scope(self.name + '_' + 'conv4_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv5_1
      with tf.variable_scope(self.name + '_' + 'conv5_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      # conv5_2
      with tf.variable_scope(self.name + '_' + 'conv5_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv5_3
      with tf.variable_scope(self.name + '_' + 'conv5_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      # fc1
      with tf.variable_scope(self.name + '_' + 'fc1') as scope:
        fc1w = tf.get_variable('fc1w', [7*7*512,4096], initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        fc1b = tf.get_variable('fc1b', [4096], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.parameters_conv += [fc1w, fc1b]

      # fc2
      with tf.variable_scope(self.name + '_' + 'fc2') as scope:
        fc2w = tf.get_variable('fc2w', [4096,4096], initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        fc2b = tf.get_variable('fc2b', [4096], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.parameters_conv += [fc2w, fc2b]
        
      # deconv0
      with tf.variable_scope(self.name + '_' + 'deconv0') as scope:
        if self.concat:
          kernel     = tf.get_variable('kernel', (3, 3, 1, 195), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        else:
          kernel     = tf.get_variable('kernel', (3, 3, 1, 64), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.deconv += [kernel, biases]
        
      # deconv1_1
      with tf.variable_scope(self.name + '_' + 'deconv1_1') as scope:
        if self.concat:
          kernel     = tf.get_variable('kernel', (3, 3, 64, 195), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        else:
          kernel     = tf.get_variable('kernel', (3, 3, 64, 64), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]

      # deconv1_2
      with tf.variable_scope(self.name + '_' + 'deconv1_2') as scope:
        if self.concat:
          kernel1      = tf.get_variable('kernel1', (3, 3, 64, 64), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
          kernel2      = tf.get_variable('kernel2', (3, 3, 64, 387), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
          biases       = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0), dtype=tf.float32)
          weight_decay = tf.mul(tf.nn.l2_loss(tf.concat(3,[kernel1,kernel2])), self.wd)
          tf.add_to_collection(self.losses, weight_decay)
          tf.add_to_collection(self.losses_decay, weight_decay)
          self.parameters_deconv += [[kernel1,kernel2], biases]
        else:
          kernel       = tf.get_variable('kernel', (3, 3, 64, 64), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
          biases       = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0), dtype=tf.float32)
          weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
          tf.add_to_collection(self.losses, weight_decay)
          tf.add_to_collection(self.losses_decay, weight_decay)
          self.parameters_deconv += [kernel, biases]

      # deconv2_1
      with tf.variable_scope(self.name + '_' + 'deconv2_1') as scope:
        if self.concat:
          kernel1      = tf.get_variable('kernel1', (3, 3, 64, 128), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
          kernel2      = tf.get_variable('kernel2', (3, 3, 64, 387), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
          biases       = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0), dtype=tf.float32)
          weight_decay = tf.mul(tf.nn.l2_loss(tf.concat(3,[kernel1,kernel2])), self.wd)
          tf.add_to_collection(self.losses, weight_decay)
          tf.add_to_collection(self.losses_decay, weight_decay)
          self.parameters_deconv += [[kernel1,kernel2], biases]
        else:
          kernel = tf.get_variable('kernel', (3, 3, 64, 128), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
          biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0), dtype=tf.float32)
          weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
          tf.add_to_collection(self.losses, weight_decay)
          tf.add_to_collection(self.losses_decay, weight_decay)
          self.parameters_deconv += [kernel, biases]
        
      # deconv2_2
      with tf.variable_scope(self.name + '_' + 'deconv2_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 128, 128), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]

      # deconv3_1
      with tf.variable_scope(self.name + '_' + 'deconv3_1') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 128, 256), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]

      # deconv3_2
      with tf.variable_scope(self.name + '_' + 'deconv3_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 256, 256), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]
        
      # deconv3_3
      with tf.variable_scope(self.name + '_' + 'deconv3_3') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 256, 256), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]

      # deconv4_1
      with tf.variable_scope(self.name + '_' + 'deconv4_1') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 256, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]

      # deconv4_2
      with tf.variable_scope(self.name + '_' + 'deconv4_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]

      # deconv4_3
      with tf.variable_scope(self.name + '_' + 'deconv4_3') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]

      # deconv5_1
      with tf.variable_scope(self.name + '_' + 'deconv5_1') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]
        
      # deconv5_2
      with tf.variable_scope(self.name + '_' + 'deconv5_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3,512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]

      # deconv5_3
      with tf.variable_scope(self.name + '_' + 'deconv5_3') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_deconv += [kernel, biases]
        
      # de_fc1
      with tf.variable_scope(self.name + '_' + 'defc1') as scope:
        fc1w = tf.get_variable('fc1w', [4096,7*7*512], initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        fc1b = tf.get_variable('fc1b', [7*7*512], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.parameters_deconv += [fc1w, fc1b]

      # de_fc2
      with tf.variable_scope(self.name + '_' + 'defc2') as scope:
        fc2w = tf.get_variable('fc2w', [4096,4096], initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        fc2b = tf.get_variable('fc2b', [4096], initializer=tf.constant_initializer(0), dtype=tf.float32)
        self.parameters_deconv += [fc2w, fc2b]
        
        

        
        
  def display_info(self, verbosity):
    """
    Display information about this model
    
    Args:
      verbosity : level of details to display
    """
    
    print('------------------------------------------------------')  
    print('This model is %s' % (self.name))
    print('------------------------------------------------------')
    if verbosity > 0:
      print('Learning rate: %0.8f -- Weight decay: %0.8f -- Cross entropy loss: %r' % (self.learning_rate , self.wd, not self.l2_loss))
      print('------------------------------------------------------')
    print('Direct connections: %r' % (self.concat))
    print('------------------------------------------------------')
    print('Edge contrast penalty: %r -- coefficient %0.5f' % (self.penalty, self.coef))
    print('------------------------------------------------------\n')
  
  
    
    
      
  def infer(self, images, inter_layer = False, arithmetic = None, debug = False):
    """
    Return saliency map from given images
    
    Args:
      images          : input images
      inter_layer     : whether we want to return the middle layer code
      arithmetic      : type of special operation on the middle layer encoding (1 is add, 2 subtract, 3 is linear combination)
      debug           : whether to return a extra value use for debug (control value)
      
    Returns:
      out             : saliency maps of the input
      control_value   : some value used to debug training
      inter_layer_out : value of the middle layer
    """

    control_value   = None
    inter_layer_out = None 

    if self.concat:
      detail      =  []
      detail_bis  =  []
      detail      += [tf.image.resize_images(images,[112,112])]
      detail_bis  += [images]
    
    # conv1_1
    with tf.variable_scope(self.name + '_' + 'conv1_1') as scope:
      conv = tf.nn.conv2d(images, self.parameters_conv[0], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[1])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      if self.concat:
        detail     += [tf.image.resize_images(norm,[112,112])]
        detail_bis += [norm]
      
    # conv1_2
    with tf.variable_scope(self.name + '_' + 'conv1_2') as scope:
      conv = tf.nn.conv2d(norm, self.parameters_conv[2], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[3])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      if self.concat:
        detail     += [tf.image.resize_images(norm,[112,112])]
        detail_bis += [norm]

    # pool1
    pool1 = tf.nn.max_pool(norm,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

    # conv2_1
    with tf.variable_scope(self.name + '_' + 'conv2_1') as scope:
      conv = tf.nn.conv2d(pool1, self.parameters_conv[4], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[5])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      if self.concat:
        detail += [norm]

    # conv2_2
    with tf.variable_scope(self.name + '_' + 'conv2_2') as scope:
      conv = tf.nn.conv2d(norm, self.parameters_conv[6], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[7])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      if self.concat:
        detail += [norm]

    # pool2
    pool2 = tf.nn.max_pool(norm,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')

    # conv3_1
    with tf.variable_scope(self.name + '_' + 'conv3_1') as scope:
      conv = tf.nn.conv2d(pool2, self.parameters_conv[8], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[9])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # conv3_2
    with tf.variable_scope(self.name + '_' + 'conv3_2') as scope:
      conv = tf.nn.conv2d(norm, self.parameters_conv[10], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[11])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # conv3_3
    with tf.variable_scope(self.name + '_' + 'conv3_3') as scope:
      conv = tf.nn.conv2d(norm, self.parameters_conv[12], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[13])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
   
    # pool3
    pool3 = tf.nn.max_pool(norm,  ksize=[1, 2, 2, 1],    strides=[1, 2, 2, 1],    padding='SAME',   name='pool3')

    # conv4_1
    with tf.variable_scope(self.name + '_' + 'conv4_1') as scope:
      conv = tf.nn.conv2d(pool3, self.parameters_conv[14], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[15])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # conv4_2
    with tf.variable_scope(self.name + '_' + 'conv4_2') as scope:
      conv = tf.nn.conv2d(norm, self.parameters_conv[16], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[17])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # conv4_3
    with tf.variable_scope(self.name + '_' + 'conv4_3') as scope:
      conv = tf.nn.conv2d(norm, self.parameters_conv[18], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[19])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # pool4
    pool4 = tf.nn.max_pool(norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],    padding='SAME', name='pool4')

    # conv5_1
    with tf.variable_scope(self.name + '_' + 'conv5_1') as scope:
      conv = tf.nn.conv2d(pool4, self.parameters_conv[20], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[21])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # conv5_2
    with tf.variable_scope(self.name + '_' + 'conv5_2') as scope:
      conv = tf.nn.conv2d(norm, self.parameters_conv[22], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[23])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # conv5_3
    with tf.variable_scope(self.name + '_' + 'conv5_3') as scope:
      conv = tf.nn.conv2d(norm, self.parameters_conv[24], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[25])
      relu = tf.nn.relu(out)
      norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # pool5
    pool5 = tf.nn.max_pool(norm,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool5')
        
    # fc1
    with tf.variable_scope(self.name + '_' + 'fc1') as scope:
      pool5_flat = tf.reshape(pool5, [self.batch_size, -1])
      fc1l       = tf.nn.bias_add(tf.matmul(pool5_flat, self.parameters_conv[26]), self.parameters_conv[27])
      fc1        = tf.nn.relu(fc1l)

    # fc2
    with tf.variable_scope(self.name + '_' + 'fc2') as scope:
      fc2l      = tf.nn.bias_add(tf.matmul(fc1, self.parameters_conv[28]), self.parameters_conv[29])
      fc2       = tf.nn.relu(fc2l)
      if inter_layer:
        inter_layer_out = fc2
      if arithmetic is not None:
        if arithmetic == 3:
          im1 = tf.squeeze(tf.split(0,self.batch_size,fc2)[0])
          im2 = tf.squeeze(tf.split(0,self.batch_size,fc2)[1])
          vec = tf.sub(im2,im1)
          liste = []
          for i in range(self.batch_size):
            liste.append(im1+i/15*vec)
          fc2 = tf.pack(liste)
        elif arithmetic == 2:
          norm = tf.sqrt(tf.reduce_sum(tf.square(fc2), 1, keep_dims=True))
          fc2  = tf.div(fc2,norm)
          im1  = tf.squeeze(tf.split(0,self.batch_size,fc2)[0])
          fc2  = tf.sub(fc2,im1)
        elif arithmetic == 1:
          norm = tf.sqrt(tf.reduce_sum(tf.square(fc2), 1, keep_dims=True))
          fc2  = tf.div(fc2,norm)
          im1  = tf.squeeze(tf.split(0,self.batch_size,fc2)[0])
          fc2  = tf.add(fc2,im1)
      
    # de-fc2
    with tf.variable_scope(self.name + '_' + 'defc2') as scope:
      fc2l = tf.nn.bias_add(tf.matmul(fc2, self.parameters_deconv[28]), self.parameters_deconv[29])
      fc2  = tf.nn.relu(fc2l)

    # de-fc1
    with tf.variable_scope(self.name + '_' + 'defc1') as scope:
      fc1l       = tf.nn.bias_add(tf.matmul(fc2, self.parameters_deconv[26]), self.parameters_deconv[27])
      fc1        = tf.nn.relu(fc1l)
      pool5_flat = tf.reshape(fc1, pool5.get_shape())
       

    # deconv5_3
    with tf.variable_scope(self.name + '_' + 'deconv5_3') as scope:
      deconv =  tf.nn.conv2d_transpose(pool5_flat, self.parameters_deconv[24], (self.batch_size,14,14,512), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[25])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      
    # deconv5_2
    with tf.variable_scope(self.name + '_' + 'deconv5_2') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[22], (self.batch_size,14,14,512), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[23])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)   

    # deconv5_1
    with tf.variable_scope(self.name + '_' + 'deconv5_1') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[20], (self.batch_size,14,14,512), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[21])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # deconv4_3
    with tf.variable_scope(self.name + '_' + 'deconv4_3') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[18], (self.batch_size,28,28,512), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[19])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # deconv4_2
    with tf.variable_scope(self.name + '_' + 'deconv4_2') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[16], (self.batch_size,28,28,512), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[17])
      relu   =  tf.nn.relu(bias)  
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      
    # deconv4_1
    with tf.variable_scope(self.name + '_' + 'deconv4_1') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[14], (self.batch_size,28,28,256), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[15])
      relu   =  tf.nn.relu(bias)  
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # deconv3_3
    with tf.variable_scope(self.name + '_' + 'deconv3_3') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[12], (self.batch_size,56,56,256), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[13])
      relu   =  tf.nn.relu(bias)
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) 

    # deconv3_2
    with tf.variable_scope(self.name + '_' + 'deconv3_2') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[10], (self.batch_size,56,56,256), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[11])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      
    # deconv3_1
    with tf.variable_scope(self.name + '_' + 'deconv3_1') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[8], (self.batch_size,56,56,128), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[9])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      
    if self.concat:
      add      = tf.concat(3,detail)
      add_bis  = tf.concat(3,detail_bis)
      if arithmetic:
        add     = tf.zeros_like(add)
        add_bis = tf.zeros_like(add_bis)
    
    # deconv2_2
    with tf.variable_scope(self.name + '_' + 'deconv2_2') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[6], (self.batch_size,112,112,128), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[7])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      if self.concat:
        norm =  tf.concat(3,[norm,add])
    
    # deconv2_1
    with tf.variable_scope(self.name + '_' + 'deconv2_1') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, tf.concat(3,self.parameters_deconv[4]), (self.batch_size,112,112,64), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[5])
      relu   =  tf.nn.relu(bias)  
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      if self.concat:
        norm = tf.concat(3,[norm,add])
                       
    # deconv1_2
    with tf.variable_scope(self.name + '_' + 'deconv1_2') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, tf.concat(3,self.parameters_deconv[2]), (self.batch_size,224,224,64), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[3])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      if self.concat:
        norm = tf.concat(3,[norm,add_bis])
      
    # deconv1_1
    with tf.variable_scope(self.name + '_' + 'deconv1_1') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.parameters_deconv[0], (self.batch_size,224,224,64), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_deconv[1])
      relu   =  tf.nn.relu(bias) 
      norm   =  tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      if self.concat:
        norm = tf.concat(3,[norm,add_bis])
      
    # deconv0
    with tf.variable_scope(self.name + '_' + 'deconv0') as scope:
      deconv =  tf.nn.conv2d_transpose(norm, self.deconv[0], (self.batch_size,224,224,1), strides= [1, 1, 1, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.deconv[1])
      relu   =  tf.sigmoid(bias)
      out = tf.squeeze(relu)
      
    if debug:
      control_value = tf.reduce_mean(relu)
      
    return out, control_value, inter_layer_out



  
  
  def loss(self, guess, labels, loss_bis = False):
    """
    Return the loss for given saliency map with corresponding ground truth
    
    Args:
      guess    :    input saliency map
      labels   :    corresponding ground truth
      loss_bis :    is it the main loss or the auxiliary one (for validation while training)
      
    Returns:
      loss_out :    the loss value
    """
    
    if self.l2_loss:
      reconstruction      = tf.reduce_sum(tf.square(guess - labels), [1,2])
      reconstruction_mean = tf.reduce_mean(reconstruction)
      if not loss_bis:
        tf.add_to_collection(self.losses, reconstruction_mean)
    else:
      guess_flat  = tf.reshape(guess,  [self.batch_size, -1])
      labels_flat = tf.reshape(labels, [self.batch_size, -1])
      zero        = tf.fill(tf.shape(guess_flat), 1e-7)
      one         = tf.fill(tf.shape(guess_flat), 1 - 1e-7)
      ret_1       = tf.select(guess_flat > 1e-7, guess_flat, zero)
      ret_2       = tf.select(ret_1 < 1 - 1e-7, ret_1, one)
      loss        = tf.reduce_mean(- labels_flat * tf.log(ret_2) - (1. - labels_flat) * tf.log(1. - ret_2))
      if not loss_bis:
        tf.add_to_collection(self.losses, loss)
      elif loss_bis:
        tf.add_to_collection(self.losses_decay, loss)
      
    if self.penalty and not loss_bis:
      labels_new   = tf.reshape(labels, [self.batch_size, 224, 224, 1])
      guess_new    = tf.reshape(guess, [self.batch_size, 224, 224, 1])
      filter_x     = tf.constant(np.array([[0,0,0] , [-1,2,-1], [0,0,0]]).reshape((3,3,1,1)), dtype=tf.float32)
      filter_y     = tf.constant(np.array([[0,-1,0] , [0,2,0], [0,-1,0]]).reshape((3,3,1,1)), dtype=tf.float32)
      gradient_x   = tf.nn.conv2d(labels_new, filter_x, [1,1,1,1], padding = "SAME")
      gradient_y   = tf.nn.conv2d(labels_new, filter_y, [1,1,1,1], padding = "SAME")
      result_x     = tf.greater(gradient_x,0)
      result_y     = tf.greater(gradient_y,0)
      keep         = tf.cast(tf.logical_or(result_x,result_y), tf.float32)  #edges

      filter_neighboor_1 = tf.constant(np.array([[0,0,0], [0,1,-1], [0,0,0]]).reshape((3,3,1)), dtype=tf.float32)
      filter_neighboor_2 = tf.constant(np.array([[0,-1,0], [0,1,0], [0,0,0]]).reshape((3,3,1)), dtype=tf.float32)
      filter_neighboor_3 = tf.constant(np.array([[0,0,0], [-1,1,0], [0,0,0]]).reshape((3,3,1)), dtype=tf.float32)
      filter_neighboor_4 = tf.constant(np.array([[0,0,0], [0,1,0], [0,-1,0]]).reshape((3,3,1)), dtype=tf.float32)
      filter_neighboor   = tf.pack([filter_neighboor_1,filter_neighboor_2,filter_neighboor_3,filter_neighboor_4], axis = 3)
      compare            = tf.square(keep * tf.nn.conv2d(guess_new, filter_neighboor, [1,1,1,1], padding = "SAME"))

      compare_m       = tf.nn.conv2d(labels_new, filter_neighboor, [1,1,1,1], padding = "SAME")
      new_compare_m   = tf.select(tf.equal(compare_m, 0), tf.ones([self.batch_size,224,224,4]), -1*tf.ones([self.batch_size,224,224,4])) #0 mean same so want to minimize and if not then diff so want to maximize
      final_compare_m = keep * new_compare_m
      
      score_ret = tf.reduce_sum(final_compare_m * compare, [1,2,3]) / (4*(tf.reduce_sum(keep,[1,2,3])+1e-7))
      score     = self.coef * tf.reduce_mean(score_ret)
      tf.add_to_collection(self.losses, score)
    
    if loss_bis:
      loss_out = tf.add_n(tf.get_collection(self.losses_decay))
    else:
      loss_out = tf.add_n(tf.get_collection(self.losses))
    
    return loss_out

  

  
  
  def train(self, loss, global_step):
    """
    Return a training step for the tensorflow graph
    
    Args:
      loss                   : loss to do sgd on
      global_step            : which step are we at
    """

    opt = tf.train.AdamOptimizer(self.learning_rate)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    variable_averages = tf.train.ExponentialMovingAverage(self.moving_avg_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')
  
    return train_op