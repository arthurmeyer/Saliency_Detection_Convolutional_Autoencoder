""" --------------------------------------------------
    author: arthur meyer
    email: arthur.meyer.38@gmail.com  
    status: final
    version: v2.0
    --------------------------------------------------"""



from __future__ import division
from __future__ import print_function
from datetime import datetime

import os
import sys
import time
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

import model



def create_model(name, batch_size, learning_rate = 0.0001, wd = 0.00001, concat = False, l2_loss = False, penalty = False, coef = 0.4, verbosity = 0):
  """
  Create a model from model.py with the given configuration
  
  Args:
    name             : name of the model (used to create a specific folder to save/load parameters)
    batch_size       : batch size
    learning_rate    : learning_rate (cross entropy is arround 100* bigger than l2)
    wd               : weight decay factor
    concat           : does this model include direct connections?
    l2_loss          : does this model use l2 loss (if not then cross entropy)
    penalty          : whether to use the edge contrast penalty
    coef             : coef for the edge contrast penalty
    verbosity        : level of details to display
    
  Returns:
    my_model         : created model
  """
  
  my_model = model.MODEL(name, batch_size, learning_rate, wd, concat, l2_loss, penalty, coef)
  my_model.display_info(verbosity)
  return my_model




def do_train(model, sess, stream_input, stream_input_aux, max_step, log_folder, mode, weight_file = None, model_to_copy = None, model_copy_is_concat = False, valid = True, dataset = None, save_copy = False):
  """
  Train the model
  
  Args:
    model                :         model to compute the score of
    sess                 :         tensorflow session
    stream_input         :         data manager
    stream_input_aux     :         data manager for auxiliary dataset (valdiation)
    max_step             :         numbers of step to train
    log_folder           :         where is the log of the model
    mode                 :         how to initialize weights
    weight_file          :         location of vgg model if pretraining
    model_to_copy        :         weights of model to copy if restoring only weight (from another model)
    model_copy_is_concat :         whether the model to copy has direct connections
    valid                :         whether to use validation
    dataset              :         dataset for the auxiliary data using during validation
    save_copy            :         whether to save a copy of the model at the end with the weight only
  """
  
  print('------------------------------------------------------')
  print('Starting training the model (number of steps is %d) ...'%(max_step))
  print('------------------------------------------------------\n')
    
  global_step       = tf.Variable(0, trainable = False)
  images, labels, _ = stream_input.get_inputs()
  guess, control, _ = model.infer(images, debug = True)
  loss              = model.loss(guess, labels)
  train_op          = model.train(loss, global_step)
  
  if valid:
    images_aux, labels_aux, _ = stream_input_aux.get_inputs()
    guess_aux, _, _           = model.infer(images_aux)
    loss_aux                  = model.loss(guess_aux, labels_aux, loss_bis = True)
    zeros                     = tf.zeros_like(labels_aux)
    ones                      = tf.ones_like(labels_aux)
    threshold = [i for i in range(255)]
    liste = []
    
    for thres in threshold:
      predicted_class = tf.select(guess_aux*255 > thres, ones, zeros)
      true_positive   = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, ones),tf.equal(labels_aux, ones)), tf.float32),[1,2])
      false_positive  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, ones),tf.equal(labels_aux, zeros)), tf.float32),[1,2])
      true_negative   = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, zeros),tf.equal(labels_aux, zeros)), tf.float32),[1,2])
      false_negative  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, zeros),tf.equal(labels_aux, ones)), tf.float32),[1,2])
      precision       = tf.reduce_sum(true_positive/(1e-8 + true_positive+false_positive))
      recall          = tf.reduce_sum(true_positive/(1e-8 + true_positive+false_negative))
      liste.append(tf.pack([precision,recall]))
    result = tf.pack(liste)   
    adaptive_threshold = (2*tf.reduce_mean(guess_aux,[0,1],keep_dims= True))
    adaptive_output = tf.select(guess_aux > adaptive_threshold, ones, zeros)
    adaptive_true_positive  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(adaptive_output, ones),tf.equal(labels_aux, ones)), tf.float32),[1,2])
    adaptive_false_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(adaptive_output, ones),tf.equal(labels_aux, zeros)), tf.float32),[1,2])
    adaptive_true_negative  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(adaptive_output, zeros),tf.equal(labels_aux, zeros)), tf.float32),[1,2])
    adaptive_false_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(adaptive_output, zeros),tf.equal(labels_aux, ones)), tf.float32),[1,2])
    adaptive_precision      = tf.reduce_sum(adaptive_true_positive / (1e-8 + adaptive_true_positive + adaptive_false_positive))
    adaptive_recall         = tf.reduce_sum(adaptive_true_positive / (1e-8 + adaptive_true_positive + adaptive_false_negative))      
    adaptive_f_measure      = tf.reduce_sum(1.3 * adaptive_precision * adaptive_recall / (1e-8 + 0.3 * adaptive_precision + adaptive_recall))    
  
  print('------------------------------------------------------')  #Initialisation of weights
  sess.run(tf.global_variables_initializer())
  if mode == 'pretrain':
    print('Loading weights from vgg file...')
    load_weights(model, sess, weight_file)
  elif mode == 'restore':
    print('Restoring from previous checkpoint...')
    sess.run(global_step.assign(int(restore_model(model, sess, log_folder))))
  elif mode == 'restore_w_only':
    print('Restoring (weights only) from model %s ...' % (model_to_copy))
    restore_weight_from(model, model_to_copy, sess, log_folder, copy_concat = model_copy_is_concat)
  elif mode == 'scratch':
    print('Initializing the weights from scratch')
  print('------------------------------------------------------')
  print('Done!')
  print('------------------------------------------------------ \n')
 
  tf.train.start_queue_runners(sess=sess)
  stream_input.start_threads(sess)
  
  if valid:
    stream_input_aux.start_threads(sess)
    if tf.gfile.Exists(log_folder + '/' + model.name +  '_validation_log'):
      tf.gfile.DeleteRecursively(log_folder + '/' + model.name +  '_validation_log')
    tf.gfile.MakeDirs(log_folder + '/' + model.name +  '_validation_log')
  
  for step in range(max_step):
    start_time = time.time()
    _, loss_value, control_value, step_b = sess.run([train_op, loss, control, tf.to_int32(global_step)])
    duration = time.time() - start_time
    
    if step % 5 == 0: #Display progress
      print ('%s: step %d out of %d, loss = %.5f (%.1f examples/sec; %.3f sec/batch)  --- control value is %.12f' % (datetime.now(), step_b, 
                                                                                                                     max_step-step+step_b, loss_value, stream_input.batch_size / duration, float(duration), control_value))
      
    if step % 1000 == 0 and step != 0 : #Save model
      save_model(model, sess, log_folder, step_b)
    
    if valid and step % 5000 == 0: #Validation 
      print('------------------------------------------------------')
      print('Doing validation ...')
      print('------------------------------------------------------ \n')
    
      loss_tot = 0
      num_iter = int(stream_input_aux.size_epoch / stream_input_aux.batch_size)     
      counter = np.zeros((256,3))
      
      for step1 in range(num_iter):
        sys.stdout.write('%d out of %d    \r' %(step1, num_iter))
        sys.stdout.flush()
        result_ret, adaptive_precision_ret, adaptive_recall_ret, adaptive_f_measure_ret, loss_value = sess.run([result, adaptive_precision, adaptive_recall, adaptive_f_measure, loss_aux])
        loss_tot += loss_value
        loss_mean = loss_tot/(step1+1)
        for i in range(255):
          for j in range(2):
            counter[i,j] += result_ret[i,j]
        counter[255,0] += adaptive_precision_ret
        counter[255,1] += adaptive_recall_ret
        counter[255,2] += adaptive_f_measure_ret
      file = open(log_folder + '/' + model.name +  '_validation_log/' + str(step_b) + ".txt" , 'w')
      file.write('model name is ' + model.name + '\n')
      file.write('number trained step is ' + str(step_b) + '\n')
      file.write('aux dataset is ' + str(dataset) + '\n')
      file.write('loss mean is ' + str(loss_mean) + '\n')
      file.write('split of dataset is valid\n')
      for i in range(256):
        precision = counter[i,0] / (num_iter * stream_input_aux.batch_size)
        recall    = counter[i,1] / (num_iter * stream_input_aux.batch_size)
        file.write('Precision %0.02f percent -- Recall %0.02f percent\n' %(precision*100, recall*100))
        if i == 255:
          f       = counter[i,2] / (num_iter * stream_input_aux.batch_size)
          file.write('fscore %0.04f\n' %(f))
        if i % 20 == 0:
          print('Precision %0.02f percent -- Recall %0.02f percent' %(precision*100, recall*100))
      file.close()
      print('\n------------------------------------------------------')
      print('Done!')
      print('------------------------------------------------------ \n')

  save_model(model, sess, log_folder, step_b) #Final save
  print('------------------------------------------------------')
  print('Save done!')
  if save_copy:
    save_weight_only(model, sess, log_folder, step_b) #Final save
    print('Saving weights onlt done!')
  print('------------------------------------------------------ \n')
    
    
    
    
def load_weights(model, sess, weight_file):
  """
  Load weights from given weight file (used to load pretrain weight of vgg model)
  
  Args:
    model            :         model to restore variable to
    sess             :         tensorflow session
    weight_file      :         weight file name
  """
    
  weights = np.load(weight_file)
  keys    = sorted(weights.keys())
  for i, k in enumerate(keys):
    if i <= 29:
      print('-- %s %s --' % (i,k))
      print(np.shape(weights[k]))
      sess.run(model.parameters_conv[i].assign(weights[k]))
     
    
     
      
def save_model(model, sess, log_path, step):
  """
  Save model using tensorflow checkpoint (also save hidden variables)
  
  Args:
    model            :         model to save variable from
    sess             :         tensorflow session
    log_path         :         where to save
    step             :         number of step at time of saving
  """
 
  path = log_path + '/' + model.name
  if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
  tf.gfile.MakeDirs(path)
  saver = tf.train.Saver()
  checkpoint_path = os.path.join(path, 'model.ckpt')
  saver.save(sess, checkpoint_path, global_step=step)
      
    
     
      
def save_weight_only(model, sess, log_path, step):
  """
  Save model but only weight (meaning no hidden variable)
  In practice use this to just transfer weights from one model to the other
  
  Args:
    model            :         model to save variable from
    sess             :         tensorflow session
    log_path         :         where to save
    step             :         number of step at time of saving
  """

  path = log_path + '/' + model.name + '_weight_only'
  if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
  tf.gfile.MakeDirs(path)
  
  variable_to_save = {}
  for i in range(30):
    name = 'conv_' + str(i)
    variable_to_save[name] = model.parameters_conv[i]
    if i in [2, 4] and model.concat:
      name = 'deconv_' + str(i)
      variable_to_save[name] = model.parameters_deconv[i][0]
      name = 'deconv_' + str(i) + '_bis'
      variable_to_save[name] = model.parameters_deconv[i][1]
    else:
      name = 'deconv_' + str(i)
      variable_to_save[name] = model.parameters_deconv[i]
    if i < 2:
      name = 'deconv_bis_' + str(i)
      variable_to_save[name] = model.deconv[i]
  saver = tf.train.Saver(variable_to_save)
  checkpoint_path = os.path.join(path, 'model.ckpt')
  saver.save(sess, checkpoint_path, global_step=step)
  
  
    
     
def restore_model(model, sess, log_path):
  """
  Restore model (including hidden variable)
  In practice use to resume the training of the same model
  
  Args
    model            :         model to restore variable to
    sess             :         tensorflow session
    log_path         :         where to save
    
  Returns:
    step_b           :         the step number at which training ended
  """
    
  path = log_path + '/' + model.name  
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(path)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    return ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
  else:
    print('------------------------------------------------------')
    print('No checkpoint file found')
    print('------------------------------------------------------ \n')
    exit()
      
    
     
          
def restore_weight_from(model, name, sess, log_path, copy_concat = False):
  """
  Restore model (excluding hidden variable)
  In practice use to train a model with the weight from another model. 
  As long as both model have architecture from the original model.py, then it works 
  Compatible w or w/o direct connections
  
  Args
    model            :         model to restore variable to
    name             :         name of model to copy
    sess             :         tensorflow session
    log_path         :         where to restore
    copy_concat      :         specify if the model to copy from also had direct connections
    
  Returns:
    step_b           :         the step number at which training ended
  """

  path = log_path + '/' + name + '_weight_only'
  
  variable_to_save = {}
  for i in range(30):
    name = 'conv_' + str(i)
    variable_to_save[name] = model.parameters_conv[i]
    if i < 2:
      if copy_concat == model.concat:
        name = 'deconv_' + str(i)
        variable_to_save[name] = model.parameters_deconv[i]
        name = 'deconv_bis_' + str(i)
        variable_to_save[name] = model.deconv[i]
    else:
      if i in [2, 4] and model.concat:
        name = 'deconv_' + str(i)
        variable_to_save[name] = model.parameters_deconv[i][0]
        if copy_concat:
          name = 'deconv_' + str(i) + '_bis'
          variable_to_save[name] = model.parameters_deconv[i][1]
      elif i in [2, 4] and not model.concat:
        name = 'deconv_' + str(i)
        variable_to_save[name] = model.parameters_deconv[i]
      else:
        name = 'deconv_' + str(i)
        variable_to_save[name] = model.parameters_deconv[i]

  saver = tf.train.Saver(variable_to_save)
  ckpt = tf.train.get_checkpoint_state(path)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    return ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
  else:
    print('------------------------------------------------------')
    print('No checkpoint file found')
    print('------------------------------------------------------ \n')
    exit()
  
  
  
  
def compute_score(model, sess, stream_input, restore_path, dataset, split, write = False, save = False):
  """
  Compute the precision recall score for a given model, with the addition of the F1 score.
  
  Args:
    model            :         model to compute the score of
    sess             :         tensorflow session
    stream_input     :         data manager
    restore_path     :         where is the restore file
    dataset          :         dataset tested
    split            :         which split (valid, test, etc.)
    write            :         whether to write the result in a file
    save             :         whether to save the resulting saliency maps
  """
   
  print('------------------------------------------------------')
  print('Computing score of the model on %s from %s ...'%(dataset, split))
  print('Write result file : %r -- Save images : %r' % (write, save))
  print('------------------------------------------------------\n')
  
  images, labels, names = stream_input.get_inputs()
  guess, _, _     = model.infer(images)

  zeros  = tf.zeros_like(labels)
  ones   = tf.ones_like(labels)
  threshold = [i for i in range(255)]
  liste = []
  for t in threshold:
    predicted_class = tf.select(guess*255 > t, ones, zeros)
    true_positive   = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, ones), tf.equal(labels, ones)), tf.float32),[1,2])
    false_positive  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, ones), tf.equal(labels, zeros)), tf.float32),[1,2])
    true_negative   = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, zeros), tf.equal(labels, zeros)), tf.float32),[1,2])
    false_negative  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, zeros), tf.equal(labels, ones)), tf.float32),[1,2])
    precision       = tf.reduce_sum(true_positive / (1e-8 + true_positive + false_positive))
    recall          = tf.reduce_sum(true_positive / (1e-8 + true_positive + false_negative))
    liste.append(tf.pack([precision,recall]))
  result = tf.pack(liste)   
  
  adaptive_threshold = 2*tf.reduce_mean(guess,[1,2], keep_dims= True)
  adaptive_output    = tf.select(guess > adaptive_threshold, ones, zeros)
  adaptive_true_positive  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(adaptive_output, ones),tf.equal(labels, ones)), tf.float32),[1,2])
  adaptive_false_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(adaptive_output, ones),tf.equal(labels, zeros)), tf.float32),[1,2])
  adaptive_true_negative  = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(adaptive_output, zeros),tf.equal(labels, zeros)), tf.float32),[1,2])
  adaptive_false_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(adaptive_output, zeros),tf.equal(labels, ones)), tf.float32),[1,2])
  adaptive_precision      = tf.reduce_sum(adaptive_true_positive / (1e-8 + adaptive_true_positive + adaptive_false_positive))
  adaptive_recall         = tf.reduce_sum(adaptive_true_positive / (1e-8 + adaptive_true_positive + adaptive_false_negative)) 
  adaptive_f_measure      = tf.reduce_sum(1.3 * adaptive_precision * adaptive_recall / (1e-8 + 0.3 * adaptive_precision + adaptive_recall))

  print('------------------------------------------------------')  #Initialisation of weights
  sess.run(tf.global_variables_initializer())
  print('Restoring from previous checkpoint...')
  ret_bis = restore_model(model, sess, restore_path)
  tf.train.start_queue_runners(sess=sess)
  stream_input.start_threads(sess)
  print('------------------------------------------------------')
  print('Done! Training ended at step %s' %(ret_bis))
  print('------------------------------------------------------ \n')
  
  if save: #Save result images
    path = restore_path + '/' + model.name  
    path += '/result_' + dataset + '/'
    if tf.gfile.Exists(path):
      tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)
  
  num_iter = int(stream_input.size_epoch / stream_input.batch_size)
  counter = np.zeros((256,3))
  
  print('------------------------------------------------------')
  for step in range(num_iter): #Compute score
    
    sys.stdout.write('%d out of %d    \r' %(step, num_iter))
    sys.stdout.flush()
    
    result_ret, adaptive_precision_ret, adaptive_recall_ret, adaptive_f_measure_ret, names_ret, images_ret, labels_ret, guess_ret = sess.run([result, adaptive_precision, adaptive_recall, adaptive_f_measure, names, images, labels, guess])
    
    for i in range(stream_input.batch_size):
      
      if save: #Save result images
        ret_path = path + names_ret[i]
        ret = np.asarray( images_ret[i]*255, dtype="int8" )
        Image.fromarray(ret, 'RGB').save(ret_path + '_' + 'im' + '.png')
        ret = np.asarray( labels_ret[i]*255, dtype="int8" )
        Image.fromarray(ret, 'P').save(ret_path + '_' + 'lab' + '.png')
        ret = np.asarray( guess_ret[i]*255, dtype="int8" )
        Image.fromarray(ret, 'P').save(ret_path + '_' + 'guess' + '.png')
      
    for i in range(255):
      for j in range(2):
        counter[i,j] += result_ret[i,j]
    counter[255,0] += adaptive_precision_ret
    counter[255,1] += adaptive_recall_ret
    counter[255,2] += adaptive_f_measure_ret
  print('------------------------------------------------------ \n')
      
  for i in range(256): 
    if i ==255:
      print('\n------------------------------------------------------')      
    precision = counter[i,0] / (num_iter * stream_input.batch_size)
    recall    = counter[i,1] / (num_iter * stream_input.batch_size)
    print('Precision %0.02f percent -- Recall %0.02f percent' %(precision*100, recall*100))
    if i ==255:
      print('fscore %0.04f' %(counter[i,2] / (num_iter*stream_input.batch_size)))
      print('------------------------------------------------------ \n')
    
  if write: #Save score
    file = open(restore_path + '/' + model.name + "/" + dataset + ".txt" , 'w')
    file.write('model name is ' + model.name + '\n')
    file.write('number trained step is ' + str(ret_bis) + '\n')
    file.write('test dataset is ' + str(dataset) + '\n')
    file.write('split of dataset is ' + str(split) + '\n')
    for i in range(255):
      file.write('Precision %0.02f percent -- Recall %0.02f percent\n' %(counter[i,0]/ (num_iter * stream_input.batch_size)*100, counter[i,1]/ (num_iter * stream_input.batch_size)*100))
    file.write('Precision %0.02f percent -- Recall %0.02f percent\n' %(counter[255,0]/ (num_iter * stream_input.batch_size)*100, counter[255,1]/ (num_iter * stream_input.batch_size)*100))
    file.write('fscore %0.04f\n' %(counter[255,2]/ (num_iter * stream_input.batch_size)))
    file.close()
    print('------------------------------------------------------')
    print('Log file written')
    print('------------------------------------------------------ \n')
    


    
def visual_tracking(model, sess, restore_path, folder_data):
  """
  Compute the tracking boundig boxe naively for each frame of the given sequence
  
  Args:
    model            :         model
    sess             :         tensorflow session
    restore_path     :         where is the restore file
    folder_data      :         the sequence location
  """
  
  print('------------------------------------------------------')
  print('Performing tracking on given sequence %s ...'%(folder_data))
  print('------------------------------------------------------\n')
  
  batch_size      = model.batch_size 
  images          = tf.placeholder(tf.float32, shape=(batch_size,224,224,3))
  guess, _, _     = model.infer(images)
  zeros           = tf.zeros_like(guess)
  ones            = tf.ones_like(guess)
  threshold       = 2*tf.reduce_mean(guess, keep_dims= True) #Adaptive threshold
  output          = tf.select(guess > threshold, ones, zeros)

  print('------------------------------------------------------')  #Initialisation of weights  
  sess.run(tf.global_variables_initializer())
  print('Restoring from previous checkpoint...')
  ret_bis = restore_model(model, sess, restore_path)
  print('------------------------------------------------------')
  print('Done! Training ended at step %s' %(ret_bis))
  print('------------------------------------------------------ \n')
  
  path = folder_data + '/results_tracking/' #Output folder
  if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
  tf.gfile.MakeDirs(path)
  tf.gfile.MakeDirs(path + 'out/')
  tf.gfile.MakeDirs(path + 'binary/')
  tf.gfile.MakeDirs(path + 'bb/')


  images_batch =  np.empty((0,224,224,3))
  index_representation = np.empty((batch_size), dtype='a1000')
  index = 0
  current = 0
  tot = len([f for f in os.listdir(folder_data + '/img/') if ".jpg" in f])
  
  print('------------------------------------------------------')
  for e in np.array([f for f in os.listdir(folder_data + '/img/') if ".jpg" in f]):
    
    current += 1
    sys.stdout.write('%d out of %d    \r' %(current, tot))
    sys.stdout.flush()
    
    im = Image.open(os.path.join(folder_data + '/img/', e)) #Read image one by one and add them to the batch
    im.load()
    w,h = im.size
    im = im.resize((224,224))
    im_a = np.asarray(im, dtype="int8" )
    images_batch = np.append(images_batch, [im_a], axis=0)
    index_representation[index] = e
    
    index += 1
    if index == batch_size: #Ready to be processed
      
      guess_ret, threshold_ret, output_ret = sess.run([guess, threshold, output], feed_dict={images : images_batch/255})
    
      for i in range(batch_size):
        ret_out = np.asarray( guess_ret[i]*255, dtype="int8" ) #Score output
        Image.fromarray(ret_out, 'P').save(path + 'out/' + index_representation[i] + '_out' + '.png')
    
        ret_out_b = np.asarray( output_ret[i]*255, dtype="int8" ) #Binary output
        Image.fromarray(ret_out_b, 'P').save(path + 'binary/' + index_representation[i] + '_binary' + '.png')
    
        im = Image.fromarray(np.asarray(images_batch[i], dtype = np.uint8), 'RGB')
        draw = ImageDraw.Draw(im) #Bounding box
        ret_mask = np.nonzero(ret_out_b)
        x0= np.amin(ret_mask,axis = 1)[1]
        y0=np.amin(ret_mask,axis = 1)[0]
        x1=np.amax(ret_mask,axis = 1)[1]
        y1=np.amax(ret_mask,axis = 1)[0]
        draw.rectangle([x0,y0,x1,y1],outline='red')
        im_b = im.resize((2*w,2*h))
        im_b.save(path + 'bb/' + index_representation[i] + '_final' + '.png')
        del draw
    
      images_batch =  np.empty((0,224,224,3))
      index_representation = np.empty((batch_size), dtype='a1000')
      index = 0  
  print('------------------------------------------------------ \n')
    
  if index != 0: #Last batch not processed
    
    number = index
    while index != batch_size:
      im = np.zeros((224,224,3))
      images_batch = np.append(images_batch, [im], axis=0)
      index +=1 
      
    guess_ret, threshold_ret, output_ret = sess.run([guess, threshold, output], feed_dict={images : images_batch/255})
    
    for i in range(number):
      ret_out = np.asarray( guess_ret[i]*255, dtype="int8" ) #Score output
      Image.fromarray(ret_out, 'P').save(path + 'out/' + index_representation[i] + '_out' + '.png')
    
      ret_out_b = np.asarray( output_ret[i]*255, dtype="int8" ) #Binary output
      Image.fromarray(ret_out_b, 'P').save(path + 'binary/' + index_representation[i] + '_binary' + '.png')
    
      im = Image.fromarray(np.asarray(images_batch[i], dtype = np.uint8), 'RGB')
      draw = ImageDraw.Draw(im) #Bounding box
      ret_mask = np.nonzero(ret_out_b)
      x0= np.amin(ret_mask,axis = 1)[1]
      y0=np.amin(ret_mask,axis = 1)[0]
      x1=np.amax(ret_mask,axis = 1)[1]
      y1=np.amax(ret_mask,axis = 1)[0]
      draw.rectangle([x0,y0,x1,y1],outline='red')
      im_b = im.resize((2*w,2*h))
      im_b.save(path + 'bb/' + index_representation[i] + '_final' + '.png')
      del draw
      
  print('------------------------------------------------------ ')
  print('Done!')
  print('------------------------------------------------------ \n')
    
    
    
        
def compute_inter(model, sess, stream_input, restore_path, dataset, split, arithmetic):
  """
  From real encoding of images, comoute new encodign and save the final results
  
  Args:
    model            :         model
    sess             :         tensorflow session
    stream_input     :         data manager
    restore_path     :         where is the restore file
    dataset          :         which dataset
    split            :         which split (valid, test, etc.)
    arithmetic  :         how to transform encoding (1 is add, 2 subtract, 3 is linear combination)
  """
   
  print('------------------------------------------------------')
  print('Computing operations on encoding of images of %s from %s ...'%(dataset, split))
  print('------------------------------------------------------')  
  if arithmetic == 1:
    print('Operation is addition')
  elif arithmetic == 2:
    print('Operation is subtraction')
  elif arithmetic == 3:
    print('Operation is linear combination')
  print('------------------------------------------------------\n')
  
  images,labels, name  = stream_input.get_inputs()
  guess, _, _          = model.infer(images, arithmetic = arithmetic)
  
  print('------------------------------------------------------')  #Initialisation of weights    
  sess.run(tf.global_variables_initializer())
  print('Restoring from previous checkpoint...')
  ret_bis = restore_model(model, sess, restore_path)
  tf.train.start_queue_runners(sess=sess)
  stream_input.start_threads(sess)
  print('------------------------------------------------------')
  print('Done! Training ended at step %s' %(ret_bis))
  print('------------------------------------------------------ \n')
  
  path = restore_path + '/' + model.name  
  path += '/results_arith_' + dataset + '_' + split + '_' + str(arithmetic) + '/'
  if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
  tf.gfile.MakeDirs(path)
  
  num_iter = int(stream_input.size_epoch / stream_input.batch_size)
  print('------------------------------------------------------')
  print('There are %d data to process in %d iterations' %(int(stream_input.size_epoch), num_iter))
  print('------------------------------------------------------ \n')
  
  for step in range(num_iter):
    
    sys.stdout.write('%d out of %d    \r' %(step, num_iter))
    sys.stdout.flush()
    
    images_ret, labels_ret, guess_ret, name_ret = sess.run([images, labels, guess, name])
    
    for i in range(model.batch_size):
      
      ret_path = path + str(i + step*model.batch_size) + '_' + str(name_ret[i])
      ret = np.asarray( images_ret[i]*255, dtype="int8" )
      Image.fromarray(ret, 'RGB').save(ret_path + '_' + 'im' + '.jpg')
      ret = np.asarray( labels_ret[i]*255, dtype="int8" )
      Image.fromarray(ret, 'P').save(ret_path + '_' + 'lab' + '.png')
      ret = np.asarray( guess_ret[i]*255, dtype="int8" )
      Image.fromarray(ret, 'P').save(ret_path + '_' + 'guess' + '.png')
      
  print('------------------------------------------------------ ')
  print('Done!')
  print('------------------------------------------------------ \n')

      
      
            
def do_nearest(model, sess, stream_input, restore_path, dataset, split, k = 4):
  """
  From encoding of images, find the nearest neighbor of each image
  
  Args:
    model            :         model
    sess             :         tensorflow session
    stream_input     :         data manager
    restore_path     :         where is the restore file
    dataset          :         which dataset
    split            :         which split (valid, test, etc.)
    k                :         number of closest
  """ 
   
  print('------------------------------------------------------')
  print('Computing the %d nearest neighbors of images of %s from %s ...'%(k, dataset, split))
  print('------------------------------------------------------\n')
  
  images, labels, name    = stream_input.get_inputs()
  guess, _, inter_feature = model.infer(images, inter_layer = True)

  num_iter     = int(stream_input.size_epoch / stream_input.batch_size)
  num_examples = num_iter * stream_input.batch_size
  dimension    = inter_feature.get_shape()[1].value
  
  index_representation = np.empty((num_examples), dtype='a1000')
  representation       = np.zeros((num_examples,dimension)) 
  closest              = np.zeros((num_examples,k+1)) 
  
  print('------------------------------------------------------')  #Initialisation of weights    
  sess.run(tf.global_variables_initializer())
  print('Restoring from previous checkpoint...')
  ret = restore_model(model, sess, restore_path)
  tf.train.start_queue_runners(sess=sess)
  stream_input.start_threads(sess)
  print('------------------------------------------------------')
  print('Done! Training ended at step %s' %(ret))
  print('------------------------------------------------------ \n')
  
  print('------------------------------------------------------')
  print('There are %d data to process in %d iterations' %(num_examples, num_iter))
  print('------------------------------------------------------ \n')
   
  for step in range(num_iter):
    
    sys.stdout.write('%d out of %d    \r' %(step, num_iter))
    sys.stdout.flush()
    
    code, name_ret = sess.run([inter_feature, name])
    for i in range(stream_input.batch_size):
      for j in range(dimension):
        representation[i + step  * stream_input.batch_size,j]   = code[i,j]
      index_representation[i + step  * stream_input.batch_size] = name_ret[i]
      
  print('------------------------------------------------------ ')
  print('Step 1: Done!')
  print('------------------------------------------------------ \n')
      
  path = restore_path + '/' + model.name  
  path += '/neighbour_' + dataset + '_' + split + '/'
  if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
  tf.gfile.MakeDirs(path)

  for i in range(num_examples):  
    
    sys.stdout.write('%d out of %d    \r' %(i, num_examples))
    sys.stdout.flush()
    
    ret = np.reshape(np.tile(representation[i],num_examples),(num_examples,dimension))
    distance = np.sum(np.square(representation-ret),axis=1)
    closest[i] = np.argsort(distance)[0:k+1]
    shutil.copy2(os.path.join(stream_input.f1 , str(index_representation[i])),path + str(index_representation[i]).split('.')[0] + '_im' + '.jpg')
    for j in range(k+1):
      if j > 0:
        shutil.copy2(os.path.join(stream_input.f1, str(index_representation[int(closest[i,j])])),path + str(index_representation[i]).split('.')[0] + '_neighbour_' + str(j) + '_' + str(index_representation[int(closest[i,j])]).split('.')[0] + '.jpg')
      
  print('------------------------------------------------------ ')
  print('Step 2: Done!')
  print('------------------------------------------------------ \n')