
# coding: utf-8

# In[1]:


import cv2
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR='/home/aaditya/Documents/pytut/train_nn'
TEST_DIR='/home/aaditya/Documents/pytut/test_nn'

IMG_SIZE = 60
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
#convnet=conv_2d(convnet,64,3,2,activation='relu')
def conv_2d(x, n, f, s=1, padding='SAME', activation='None' ):
    if activation=='relu':
        return tf.contrib.layers.conv2d(x, n, f, s, padding=padding)
    elif activation == 'None':
        return tf.contrib.layers.conv2d(x, n, f, s, activation_fn=None)
def max_pool_2d(x, f):
    return tf.contrib.layers.max_pool2d(x, f)
def avg_pool_2d(x, f):
    return tf.contrib.layers.avg_pool2d(x, f)
def flatten(x):
    return tf.contrib.layers.flatten(x)
def fully_connected(x, f, bias=False):
    return tf.layers.dense(x, f, use_bias=bias)


def label_img (img):
    word_label=img.split('.')[-3]
    if word_label=='_0':
        return [1,0,0]
    elif word_label=='_1':
        return[1,0,0]
    elif word_label=='_2':
        return[1,0,0]
    elif word_label=='_3':
        return[1,0,0]
    elif word_label=='_4':
        return[0,1,0]
    elif word_label=='_5':
        return[0,1,0]
    elif word_label=='_6':
        return[0,1,0]
    elif word_label=='_7':
        return[0,0,1]
    elif word_label=='_8':
        return[0,0,1]
    elif word_label=='_9':
        return[0,0,1]
def label_img1 (img):
    word_label=img.split('.')[-2]
    if word_label=='0':
        return [1,0,0]
    elif word_label=='1':
        return[1,0,0]
    elif word_label=='2':
        return[1,0,0]
    elif word_label=='3':
        return[1,0,0]
    elif word_label=='4':
        return[0,1,0]
    elif word_label=='5':
        return[0,1,0]
    elif word_label=='6':
        return[0,1,0]
    elif word_label=='7':
        return[0,0,1]
    elif word_label=='8':
        return[0,0,1]
    elif word_label=='9':
        return[0,0,1]


def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        img =cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
        
                            

    shuffle(training_data)
    #np.save('train_data.npy',training_data)
    return training_data

def process_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        label=label_img1(img)
        img_num = img.split('.')[0]
        path=os.path.join(TEST_DIR,img)
        img =cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),np.array(label)])
        

    shuffle(testing_data)
    #np.save('test_data.npy',testing_data)
    return testing_data

train_data = create_train_data()
test_data=process_test_data()



#import tflearn
#from tflearn.layers.conv import conv_2d,max_pool_2d, avg_pool_2d
#from tflearn.layers.core import input_data,dropout,fully_connected, flatten
#from tflearn.layers.estimator import regression

x = tf.placeholder('float', [None, 60, 60, 1])
y = tf.placeholder('float')

convnet=conv_2d(x,32,7,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet1=convnet

print(convnet.get_shape().as_list())  #30

convnet=conv_2d(convnet,32,3,activation='relu')
convnet=conv_2d(convnet,32,3,activation='relu')
convnet=convnet1+convnet
convnet1=convnet

print(convnet.get_shape().as_list())  #30

convnet=conv_2d(convnet,32,3,activation='relu')
convnet=conv_2d(convnet,32,3,activation='relu')
convnet=convnet1+convnet

print(convnet.get_shape().as_list())  #30

convnet=conv_2d(convnet,64,3,2,activation='relu')
convnet1=convnet
print(convnet.get_shape().as_list())  #15

convnet=conv_2d(convnet,64,3,activation='relu')
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=convnet1+convnet
convnet1 = convnet

print(convnet.get_shape().as_list())  #15

convnet=conv_2d(convnet,64,3, padding='valid', activation='relu')
convnet1=convnet
print(convnet.get_shape().as_list())  #13

convnet=conv_2d(convnet,64,3,activation='relu')
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=convnet1+convnet
convnet1 = convnet

print(convnet.get_shape().as_list())  #13

convnet=conv_2d(convnet,128,3, padding='valid', activation='relu')
convnet1=convnet
print(convnet.get_shape().as_list())  #11

convnet=conv_2d(convnet,128,3,activation='relu')
convnet=conv_2d(convnet,128,3,activation='relu')
convnet=convnet1+convnet
convnet1=convnet

print(convnet.get_shape().as_list())

convnet=conv_2d(convnet,128,3, padding='valid', activation='relu')
convnet1=convnet
print(convnet.get_shape().as_list())  #9

convnet=conv_2d(convnet,128,3,activation='relu')
convnet=conv_2d(convnet,128,3,activation='relu')
convnet=convnet1+convnet
convnet1=convnet

print(convnet.get_shape().as_list())

convnet=conv_2d(convnet,128,3, padding='valid', activation='relu')
convnet1=convnet
print(convnet.get_shape().as_list())  #7

convnet=conv_2d(convnet,128,3,activation='relu')
convnet=conv_2d(convnet,128,3,activation='relu')
convnet=convnet1+convnet
convnet1=convnet

print(convnet.get_shape().as_list())

convnet=conv_2d(convnet,256,3, padding='valid', activation='relu')
convnet1=convnet
print(convnet.get_shape().as_list())  #5

convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=convnet1+convnet
convnet1=convnet

print(convnet.get_shape().as_list())

convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=convnet1+convnet
convnet1=convnet

print(convnet.get_shape().as_list())

convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=convnet1+convnet

print(convnet.get_shape().as_list())

convnet=avg_pool_2d(convnet,convnet.get_shape().as_list()[1])

print(convnet.get_shape().as_list())

convnet=flatten(convnet)
prediction=fully_connected(convnet,3, bias=False)

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

train = train_data[:]
test = train_data[:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

hm_epochs = 4
batch_size = 20

saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'car-')
save_path = save_path + 'test-2.ckpt'

sess = tf.Session()
# OLD:
#sess.run(tf.initialize_all_variables())
# NEW:
start_time = time.time()

#sess.run(tf.global_variables_initializer())
saver.restore(sess, "/home/aaditya/Documents/pytut/checkpoints/car-test-2.ckpt.index")

best_acc_valid = 0
print(len(X))
for epoch in range(hm_epochs):
    epoch_loss = 0
    
    for _ in range(50):
        epoch_x, epoch_y = X[_*batch_size:(_+1)*batch_size], Y[_*batch_size:(_+1)*batch_size]
        #print(epoch_x.shape)
        n, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
        epoch_loss += c
    
    acc_valid = sess.run(accuracy, feed_dict={x: test_x, y: test_y})

    if acc_valid > best_acc_valid:
        best_acc_valid = acc_valid
        #save_path = save_path + '-' + str(epoch) + '-' + str(int(acc_valid))
        saver.save(sess=sess, save_path=save_path)
        improved_str = '*'
    else:
        improved_str = ''
    
    print('Epoch', epoch,'/',hm_epochs, 'loss: ', epoch_loss, '. Validation accuracy: ', acc_valid, improved_str)

end_time = time.time()
time_dif = end_time - start_time

print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
                  
sess.close()      
##validation_set=({'input':test_x},{'targets':test_y}),


# In[ ]:




