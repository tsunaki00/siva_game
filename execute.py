#!/usr/bin/env python
import tensorflow as tf
import sys
import numpy as np
from random import randrange
import rle_python_interface

class Game:

  def __init__(self) :
    if len(sys.argv) < 2:
      print ('Usage:', sys.argv[0], 'rom_file', 'core_file')
      sys.exit()
    self.rle = rle_python_interface.RLEInterface()
    self.rle.setInt('random_seed', 123)
  
  
  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

  
  def bias_variable(self, shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

  def conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

  
  def createNetwork(self, action_size):
    # network weights
    n = 256
    W_conv1 = self.weight_variable([56, 64, 4, 16])
    b_conv1 = self.bias_variable([16])
    W_conv2 = self.weight_variable([28, 32, 64, 32])
    b_conv2 = self.bias_variable([32])
    W_conv3 = self.weight_variable([7, 8, 32, 16])
    b_conv3 = self.bias_variable([16])
    W_fc1 = self.weight_variable([n, action_size])
    b_fc1 = self.bias_variable([action_size])
    s = tf.placeholder("float", [None, 224, 256, 4])
    # hidden layers
    h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = self.max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = self.max_pool_2x2(h_conv2)
    h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = self.max_pool_2x2(h_conv3)
    
    h_conv3_flat = tf.reshape(h_pool3, [-1, n])
    readout = tf.matmul(h_conv3_flat, W_fc1) + b_fc1
    return s, readout 

  def trainNetwork(self,  s, readout, sess):
    minimal_actions = self.rle.getMinimalActionSet()
    a = tf.placeholder("float", [None, len(minimal_actions)])
    y = tf.placeholder("float", [None, 1])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    checkpoint = tf.train.get_checkpoint_state("./saved_networks/checkpoints")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    # Play 100 episodes
    for episode in xrange(100):
      total_reward = 0
      act_count = 1
      actions = []
      rewards = []
      images = []
      while not self.rle.game_over():
        if episode >= 1 :
          readout_t = readout.eval(feed_dict = {s : [self.rle.getScreenRGB()]})[0]
          print(readout_t)
          action_index = np.argmax(readout_t)
        else :
          action_index = randrange(len(minimal_actions))

        action = minimal_actions[action_index]
        # Apply an action and get the resulting reward
        action_array = np.zeros(len(minimal_actions))
        action_array[action_index] = 1
        actions.append(action_array)
        reward = self.rle.act(action)
        total_reward += reward
        array = np.zeros(1)
        array[0] = float(reward)
        rewards.append(array)
        images.append(self.rle.getScreenRGB())
        print ('Episode', episode, 'REWARD:', reward, 'Action', action)
  
        if act_count % 50  == 0 :
          train_step.run(feed_dict = {
            a : actions,
            y : rewards,
            s : images
          })
          actions = []
          rewards = []
          images = []


        act_count += 1
    
      print ('Episode', episode, 'ended with score:', total_reward)
      saver.save(sess, 'saved_networks/model-dqn', global_step = episode)
      self.rle.reset_game()


  def playGame(self):
    USE_SDL = True
    if USE_SDL:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        self.rle.setBool('sound', False) # Sound doesn't work on OSX
      #elif sys.platform.startswith('linux'):
      #  rle.setBool('sound', True)
      self.rle.setBool('display_screen', True)
    #  rle.setBool('two_players', True)
    self.rle.loadROM(sys.argv[1], sys.argv[2])

    # Get the list of legal actions
    minimal_actions = self.rle.getMinimalActionSet()
    #minimal_actions = rle.getAllActionSet()

    sess = tf.InteractiveSession()
    s, readout = self.createNetwork(len(minimal_actions))
    self.trainNetwork(s, readout, sess)


if __name__ == '__main__' :
  game = Game()
  game.playGame()