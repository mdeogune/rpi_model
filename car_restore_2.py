
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import numpy as np
import time
from datetime import timedelta
import os
from random import shuffle
from tqdm import tqdm
IMG_SIZE = 60


# In[2]:


sess = tf.Session()

new_saver = tf.train.import_meta_graph('/home/aaditya/Documents/pytut/checkpoints/car-test-3.ckpt.meta')
new_saver.restore(sess, '/home/aaditya/Documents/pytut/checkpoints/car-test-3.ckpt')


# In[3]:


graph = tf.get_default_graph()
x = graph.get_tensor_by_name('x:0')
y = graph.get_tensor_by_name('y:0')


# In[6]:


img =cv2.imread('/home/aaditya/Documents/pytut/test_nn/ab.jpg',cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
x_=np.array(img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
pre_label = graph.get_tensor_by_name('ArgMax:0')
digit = sess.run(pre_label, feed_dict={x:x_})
print(digit[0])


# In[ ]:




