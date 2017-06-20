import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LEN = 1000

def sampleData(size, length = 100):
  # mean = 4, std = 1.5
  data = [(sorted(np.random.normal(4, 1.5, length))) for _ in range(size)]
  return np.array(data)

def randomData(size, length = 100):
  data = [(np.random.random(length)) for _ in range(size)]
  return np.array(data)

def preprocessData(data):
  return [[np.mean(vec), np.std(vec)] for vec in data]

def createGenerativeNetwork():
  ''' Generative Network '''
  with tf.name_scope('Generative') as scope:
    z = tf.placeholder(tf.float32, shape = [None, LEN], name = 'noise')
    Params = []
    
    with tf.name_scope('Layer1') as scope:
      w = tf.Variable(tf.random_normal([LEN, 32]))
      b = tf.Variable(tf.zeros([1, 32]) + 0.1)
      out = tf.nn.relu(tf.add(tf.matmul(z, w), b))
      Params.append(w)
      Params.append(b)
    
    with tf.name_scope('Layer2') as scope:
      w = tf.Variable(tf.random_normal([32, 32]))
      b = tf.Variable(tf.zeros([1, 32]) + 0.1)
      out = tf.nn.sigmoid(tf.add(tf.matmul(out, w), b))
      Params.append(w)
      Params.append(b)
    
    with tf.name_scope('Layer3') as scope:
      w = tf.Variable(tf.random_normal([32, LEN]))
      b = tf.Variable(tf.zeros([1, LEN]) + 0.1)
      out = tf.add(tf.matmul(out, w), b)
      Params.append(w)
      Params.append(b)

  return (z, Params, out)

def createDiscriminatorNetwork():
  ''' Discriminator Network '''
  with tf.name_scope('Discriminator') as scope:
    Params = []
  
    with tf.name_scope('Layer1') as scope:
      w = tf.Variable(tf.random_normal([2, 32]), name = 'D_W')
      b = tf.Variable(tf.zeros([1, 32]) + 0.1, name = 'D_B')
      out = tf.nn.relu(tf.add(tf.matmul(x, w), b))
      Params.append(w)
      Params.append(b)
    
    with tf.name_scope('Layer2') as scope:
      w = tf.Variable(tf.random_normal([32, 32]), name = 'D_W2')
      b = tf.Variable(tf.zeros([1, 32]) + 0.1, name = 'D_B2')
      out = tf.nn.sigmoid(tf.add(tf.matmul(out, w), b))
      Params.append(w)
      Params.append(b)
    
    with tf.name_scope('Layer3') as scope:
      w = tf.Variable(tf.random_normal([32, 1]), name = 'D_W3')
      b = tf.Variable(tf.zeros([1, 1]) + 0.1, name = 'D_B3')
      out = tf.add(tf.matmul(out, w), b)
      Params.append(w)
      Params.append(b)
    
  return (Params, out)

def createGenerativeAdversarialNetwork(GOutput3):
  ''' Generative Adversarial Network '''
  with tf.name_scope('GenerativeAdversarial') as scope:
    Mean = tf.reduce_mean(GOutput3, 1)
    MeanT = tf.transpose(tf.expand_dims(Mean, 0))
    Std = tf.sqrt(tf.reduce_mean(tf.square(GOutput3 - MeanT), 1))
    [MeanT, tf.transpose(tf.expand_dims(Std, 0))]
    Data = tf.concat([MeanT, tf.transpose(tf.expand_dims(Std, 0))], 1)  # mean and std of GN out
    GANParams = []
    
    with tf.name_scope('Layer1') as scope:
      w = tf.Variable(tf.random_normal([2, 32]), name = 'GAN_W1')
      b = tf.Variable(tf.zeros([1, 32]), name = 'GAN_B1')
      out = tf.nn.relu(tf.add(tf.matmul(Data, w), b))
      GANParams.append(w)
      GANParams.append(b)
    
    with tf.name_scope('Layer2') as scope:
      w = tf.Variable(tf.random_normal([32, 32]), name = 'GAN_W2')
      b = tf.Variable(tf.zeros([1, 32]), name = 'GAN_B2')
      out = tf.nn.sigmoid(tf.add(tf.matmul(out, w), b))
      GANParams.append(w)
      GANParams.append(b)
    
    with tf.name_scope('Layer3') as scope:
      w = tf.Variable(tf.random_normal([32, 1]), name = 'GAN_W3')
      b = tf.Variable(tf.zeros([1, 1]) + 0.1, name = 'GAN_B3')
      out = tf.add(tf.matmul(out, w), b)
      GANParams.append(w)
      GANParams.append(b)
  
  return (GANParams, out)

if __name__ == '__main__':
  x = tf.placeholder(tf.float32, shape = [None, 2], name = 'feature')  # Mean and std
  y = tf.placeholder(tf.float32, shape = [None, 1], name = 'label')
  
  z, GParams, Gout = createGenerativeNetwork()
  DParams, Dout = createDiscriminatorNetwork()
  GANParams, out = createGenerativeAdversarialNetwork(Gout)
  
  ''' Loss Functions '''
  DLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dout, labels = y))
  GLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = out, labels = y))

  ''' Optimizer '''
  DOpertimizer = tf.train.GradientDescentOptimizer(0.01).minimize(DLoss, global_step = tf.Variable(0), var_list = DParams)
  GOpertimizer = tf.train.GradientDescentOptimizer(0.05).minimize(GLoss, global_step = tf.Variable(0), var_list = GParams)
 
  ''' Train '''
  dLossHistory = []
  gLossHistory = []
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(100):
      # Train discrimination network
      for _ in range(100):
        real = sampleData(100, length = LEN)
        noise = randomData(100, length = LEN)
        generate = sess.run(Gout, feed_dict = {z:noise}) # generate samples
        xBatch = preprocessData(list(real) + list(generate))
        yBatch = [[1] for _ in range(len(real))] + [[0] for _ in range(len(generate))]
        dLossValue, _ = sess.run([DLoss, DOpertimizer], feed_dict = {x:xBatch, y:yBatch})
        dLossHistory.append(dLossValue)
      
      # Train generative network
      dParamValues = sess.run(DParams)
      for i, v in enumerate(GANParams):
        sess.run(v.assign(dParamValues[i]))
      for _ in range(100):
        noise = randomData(100, length = LEN)
        gLossValue, _ = sess.run([GLoss, GOpertimizer], feed_dict = {z:noise, y:[[1] for _ in range(len(noise))]})
        gLossHistory.append(gLossValue)
      
      if step % 10 == 0 or step + 1 == 100:
        noise = randomData(1, length = LEN)
        generate = sess.run(Gout, feed_dict = {z:noise})
        print "[%4d] GAN-d-loss: %.12f GAN-g-loss: %.12f generate-mean: %.4f generate-std: %.4f" % (step, dLossValue, gLossValue, generate.mean(), generate.std())

  plt.figure(1)

  plt.subplot(211)
  plt.plot(dLossHistory)
  plt.subplot(212)
  plt.plot(gLossHistory,c="g")
  plt.savefig('./M1.png')

  plt.figure(2)
  real = sampleData(1,length = LEN)
  (data, bins) = np.histogram(real[0])
  plt.plot(bins[:-1], data, c="g")

  (data, bins) = np.histogram(noise[0])
  plt.plot(bins[:-1], data, c="b")

  (data, bins) = np.histogram(generate[0])
  plt.plot(bins[:-1], data, c="r")
  plt.savefig('./M2.png')
