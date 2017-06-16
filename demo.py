""" --------------------------------------------------
    author: arthur meyer
    email: arthur.meyer.38@gmail.com  
    status: final
    version: v2.0
    --------------------------------------------------"""



import os 
import sys
import numpy as np
import tensorflow as tf

import input
import operations



HEIGHT     = 224
WIDTH      = 224
BATCH_SIZE = 16
STEPS      = 100*1000
PATH       = os.path.abspath(__file__).split('/demo.py')[0]
LOG_FOLDER = PATH + '/log'

L_R        = 0.0001
WD         = 0.00001
VERB       = 0

MODEL_NAME     = 'BDE'
PHASE          = 'test'
DATASET        = 'ecssd'
AUX            = 'msra10k'

 

def display_warning(name, specific = None):
  """
  Display warning message if wrong parameter
  
  Args:
    name     :   name of the parameter where the warning occurs
    specific :   may include a specific message to display
  """
  
  print('------------------------------------------------------')
  print('---------------------- WARNING -----------------------')
  print('------------------------------------------------------')
  print('--------- invalid argument for parameter -%s ---------' % (name))
  print('------------------------------------------------------')
  print(specific)
  print('------------------------------------------------------\n')
  exit()

  
  
  
def dataset_config():
  """
  Return the approriate input manager
    
  Returns:
    handler     : manager of data from the class input.py for the main data stream
    handler_bis : manager of data from the class input.py for the validation data stream that is only use during training
  """

  if PHASE == 'train':
    folder_im = PATH + "/dataset_split/train/images/"
    folder_lab = PATH + "/dataset_split/train/labels/"
    handler = input.handler(HEIGHT, WIDTH, BATCH_SIZE, folder_im, folder_lab, random = True)
    
    print('------------------------------------------------------')
    print('Queueing data ....')
    print('------------------------------------------------------')
    print('from TRAINING dataset -- auxiliary dataset is %s' % (AUX.upper()))
    print('------------------------------------------------------')
    print('Height: %d -- width: %d -- batch size: %d -- log folder: %s' % (HEIGHT, WIDTH, BATCH_SIZE, LOG_FOLDER))
    print('------------------------------------------------------\n')
    
  elif PHASE == 'valid' or PHASE == 'test':
    folder_im = PATH + "/dataset_split/" + PHASE + "/" + DATASET + "/images/"
    folder_lab = PATH + "/dataset_split/" + PHASE + "/" + DATASET + "/labels/"
    handler = input.handler(HEIGHT, WIDTH, BATCH_SIZE, folder_im, folder_lab, random = False)
    
    print('------------------------------------------------------')
    print('Queueing data ....')
    print('------------------------------------------------------')
    print('Dataset is %s -- from split %s' % (DATASET.upper(), PHASE.upper()))
    print('------------------------------------------------------')
    print('Height: %d -- width: %d -- batch size: %d -- log folder: %s' % (HEIGHT, WIDTH, BATCH_SIZE, LOG_FOLDER))
    print('------------------------------------------------------\n')
    
  folder_im = PATH + "/dataset_split/valid/" + AUX + "/images/"
  folder_lab = PATH + "/dataset_split/valid/" + AUX + "/labels/"
  handler_bis = input.handler(HEIGHT, WIDTH, BATCH_SIZE, folder_im, folder_lab, random = False)
    
  return handler, handler_bis




def model_config():
  """
  Return the approriate model
  
  Returns:
    model : create model with the appropriate configuration, B is for baseline BD for baseline with direct connections and BDE for baseline with direct connections and edge contrast penalty
  """
  
  if MODEL_NAME == 'B':
    model = operations.create_model('VGG_CE_noDetails_further'         , BATCH_SIZE, learning_rate = L_R, wd = WD, concat = False, l2_loss = False, penalty = False,            verbosity = VERB)
    
  elif MODEL_NAME == 'BD':
    model  = operations.create_model('VGG_CE_Details_c'                , BATCH_SIZE, learning_rate = L_R, wd = WD, concat = True,  l2_loss = False, penalty = False,            verbosity = VERB)
    
  elif MODEL_NAME == 'BDE':         
    model  = operations.create_model('VGG_CE_Details_new_from_pretrain', BATCH_SIZE, learning_rate = L_R, wd = WD, concat = True,  l2_loss = False, penalty = True, coef = 0.8, verbosity = VERB)
    
  elif MODEL_NAME == 'new':         
    model  = operations.create_model('test', BATCH_SIZE, learning_rate = L_R, wd = WD, concat = True,  l2_loss = False, penalty = True, coef = 0.8, verbosity = VERB)
    
  return model




def do_operation(sess, model):
  """
  Do the wanted operation with the model given in parameter

  Args:
    sess             : tensorflow session
    model            : configured model
  """
  
  if OPERATION   == 'train':
    stream_input, stream_input_bis = dataset_config()
    operations.do_train(model, sess, stream_input, stream_input_bis, STEPS, LOG_FOLDER, INITIALISATION, weight_file = W_FILE, model_to_copy = M_COPY, model_copy_is_concat = C_C, valid = VALID, dataset = AUX, save_copy = S_COPY)
    
  elif OPERATION == 'score':
    stream_input, _ = dataset_config()
    operations.compute_score(model, sess, stream_input, LOG_FOLDER, DATASET, PHASE, write = WRITE, save = SAVE)
  
  elif OPERATION == 'tracking':
    operations.visual_tracking(model, sess, LOG_FOLDER, PATH + '/experiments/visual_tracking/' + TRACKING) 
  
  elif OPERATION == 'infer':
    stream_input, _ = dataset_config()
    operations.compute_inter(model, sess, stream_input, LOG_FOLDER, DATASET, PHASE, ARITH_TYPE)
  
  elif OPERATION == 'nearest':
    stream_input, _ = dataset_config()
    operations.do_nearest(model, sess, stream_input, LOG_FOLDER, DATASET, PHASE) 

  elif OPERATION == 'void':
    _, _ = dataset_config()
    


    
if __name__ == '__main__':
  
  sess = tf.Session()
  l = sys.argv[1:]
  main_flag = False
  
  #PARSING Set the global variable with appropriate value
  for i in range(len(l)): 
    
    if l[i] == '-batch':
      try:
        BATCH_SIZE = int(l[i+1])
      except Exception:
        display_warning('batch')
        
    if l[i] == '-step':
      try:
        STEPS = int(l[i+1])
      except Exception:
        display_warning('step')

    if l[i] == '-height':
      try:
        HEIGHT = int(l[i+1])
      except Exception:
        display_warning('height')
        
    if l[i] == '-width':
      try:
        WIDTH = int(l[i+1])
      except Exception:
        display_warning('width')
    
    if l[i] == '-lr':
      try:
        L_R = float(l[i+1])
      except Exception:
        display_warning('lr')
        
    if l[i] == '-wd':
      try:
        WD = float(l[i+1])
      except Exception:
        display_warning('wd')

    if l[i] == '-m':
      if l[i+1] in ['B', 'BD', 'BDE', 'new']:
        MODEL_NAME = l[i+1]
      else:
        display_warning('m')
        
    if l[i] == '-d':
      if l[i+1] in ['msra10k', 'ecssd', 'dutomron']:
        DATASET = l[i+1]
      else:
        display_warning('d')
      
    if l[i] == '-p':
      if l[i+1] in ['train', 'test', 'valid']:
        PHASE = l[i+1]
      else:
        display_warning('p')
        
    if l[i] == '-o':
      if l[i+1] in ['train', 'score', 'infer', 'nearest', 'tracking', 'void']:
        OPERATION = l[i+1]
        main_flag = True
        
        if l[i+1] == 'train':
          global W_FILE
          W_FILE = 'vgg_weight/vgg16_weights.npz' 
          global M_COPY
          M_COPY = 'saliency_VGG_CE_noDetails'
          global VALID
          VALID = True
          global S_COPY
          S_COPY = False
          global C_C
          C_C = False
          
          for j in range(len(l)):
            if l[j] == '-w_file':
              W_FILE = l[j+1]
            elif l[j] == '-m_copy':
              M_COPY = l[j+1]
            elif l[j] == '-no_valid':
              VALID = False
            elif l[j] == '-s_copy':
              S_COPY = True
            elif l[j] == '-c_c':
              C_C = True
            elif l[j] == '-aux':
              if l[j+1] in ['msra10k', 'ecssd', 'dutomron']:
                AUX = l[j+1]
              else:
                display_warning('aux')
              
          flag = False
          for j in range(len(l)):
            if l[j] == '-init':
              if flag:
                display_warning('init', specific ='Multiple arguments')
              else:
                if l[j+1] in ['scratch', 'restore_w_only', 'restore', 'pretrain']:
                  global INITIALISATION
                  INITIALISATION = l[j+1]
                  flag = True
                else:
                  display_warning('init', specific ='Value incorrect')
          if not flag:
            display_warning('init', specific ='No initialization specified')
          
        if l[i+1] == 'score':
          global WRITE
          WRITE = False
          global SAVE
          SAVE = False
          for e in l:
            if e == '-write':
              WRITE = True
            elif e == '-save':
              SAVE = True
        
        if l[i+1] == 'tracking':
          flag = False
          for j in range(len(l)):
            if l[j] == '-video':
              if flag:
                display_warning('video', specific ='Multiple arguments')
              else:
                if l[j+1] in ['BlurBody', 'Dog', 'Girl', 'Gym']:
                  global TRACKING 
                  TRACKING = l[j+1]
                  flag = True
                else:
                  display_warning('video', specific ='Name incorrect')
          if not flag:
            display_warning('tracking', specific ='No video name specified')
                    
        if l[i+1] == 'infer':
          flag = False
          for j in range(len(l)):
            if l[j] == '-type':
              if flag:
                display_warning('type', specific ='Multiple arguments')
              else:
                if l[j+1] in ['1', '2', '3']:
                  global ARITH_TYPE 
                  ARITH_TYPE = int(l[j+1])
                  flag = True
                else:
                  display_warning('type', specific ='Type incorrect')
          if not flag:
            display_warning('infer', specific ='No type specified')
            
      else:
        display_warning('o')

  if main_flag:
    if OPERATION == 'train':
      PHASE = 'train'
      VERB = 1
    model = model_config()
    do_operation(sess, model)
  else:
    print('Nothing to be done')