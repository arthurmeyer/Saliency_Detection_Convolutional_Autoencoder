""" --------------------------------------------------
    author: arthur meyer
    email: arthur.meyer.38@gmail.com  
    status: final
    version: v2.0
    --------------------------------------------------"""



from __future__ import division
import os
import threading
import numpy as np
import tensorflow as tf
from PIL import Image



class handler(object):
  """ 
  This class run a thread that queue data
  Overall this class is responsible for managing the flow of data
  """
    
  def __init__(self, hight, width, batch_size, folder_image, folder_label, format_image = '.jpg' , random = True):
    """
    Args:
      hight             :         hight of samples
      width             :         width of samples
      batch_size        :         batch size
      folder_image      :         the folder where the images are
      folder_label      :         the folder where the ground truth are
      format_image      :         format of images (usually jpg)
      random            :         is the queue shuffled (for training) or not (FIFO for test related tasks)
    """  

    self.hight           =       hight
    self.width           =       width
    self.batch_size      =       batch_size
    self.image           =       np.array([f for f in os.listdir(folder_image) if format_image in f])
    self.f1              =       folder_image
    self.f2              =       folder_label
    self.size_epoch      =       len(self.image)
    if random:
      self.queue           =       tf.RandomShuffleQueue(shapes=[(self.hight,self.width,3), (self.hight,self.width), []],dtypes=[tf.float32, tf.float32, tf.string],capacity=16*self.batch_size, min_after_dequeue=8*self.batch_size)
    else:
      self.queue           =       tf.FIFOQueue(shapes=[(self.hight,self.width,3), (self.hight,self.width), []],dtypes=[tf.float32, tf.float32, tf.string],capacity=16*self.batch_size)
    self.image_pl        =       tf.placeholder(tf.float32, shape=(batch_size,hight,width,3))
    self.label_pl        =       tf.placeholder(tf.float32, shape=(batch_size,hight,width))
    self.name_pl         =       tf.placeholder(tf.string, shape=(batch_size))
    self.enqueue_op      =       self.queue.enqueue_many([self.image_pl, self.label_pl, self.name_pl])

      
    
  def get_inputs(self):
    """
    Getter for the data in the queue
    
    Returns:
      A tensor of size 'self.batch_size' of data 
    """
    
    return self.queue.dequeue_many(self.batch_size)


  
  def start_threads(self, sess):
    """
    Start the thread where the data is put into the queue 
    
    Args:
      sess : the context for the thread, here a tensorflow session
      
    Returns:
      t    : the thread started
    """
    
    t = threading.Thread(target=self._thread_main, args=(sess, ))
    t.daemon = True
    t.start()
    return t
  
  
  
  def _thread_main(self, sess):
    """
    The main thread where data is queued
    
    Args:
      sess : the context for the thread, here a tensorflow session
    """
    
    for images, labels, names in self._data_iterator():
      sess.run(self.enqueue_op, feed_dict = {self.image_pl : images, self.label_pl : labels, self.name_pl : names})

      
      
  def _data_iterator(self):
    """
    The iterator on the data managed by the class. Here images are read and delivered to the queue
    """
    
    while True: #Main loop where each epoch is shuffled
      
      batch_index = 0
      index = np.arange(0, self.size_epoch)
      np.random.shuffle(index)
      shuffled_image = self.image[index]
      
      while batch_index + self.batch_size <= self.size_epoch: #Loop on one epoch
        
        images_names =  shuffled_image[batch_index : batch_index + self.batch_size]
        batch_index  += self.batch_size
        images_batch =  np.empty((0,self.hight,self.width,3))
        label_batch  =  np.empty((0,self.hight,self.width))
        
        for f in images_names: #Loop on one batch
          
          im = Image.open(os.path.join(self.f1, f)) #First the image
          im.load()
          im = im.resize((self.width ,self.hight ))
          im = np.asarray(im, dtype="int8" )
          if len(np.shape(im)) != 3 : #If not 3 channels then warning is displayed
            print('----- WARNING -----')
            print('This image is not in RGB format:')
            print(f)
          images_batch = np.append(images_batch, [im], axis=0)
          
          im = Image.open(os.path.join(self.f2, f.split('.')[0] + '.png')) #Then the ground truth
          im.load()
          im = im.resize((self.width ,self.hight ))
          im = np.asarray(im, dtype="int16" )
          label_batch = np.append(label_batch, [im], axis=0)
          
        index = np.arange(0, self.batch_size)
        np.random.shuffle(index)
        images_batch_shuffled = images_batch[index]
        label_batch_shuffled  = label_batch[index]
        names_batch_shuffled  = images_names[index]
        
        yield images_batch_shuffled/255, label_batch_shuffled/255, names_batch_shuffled