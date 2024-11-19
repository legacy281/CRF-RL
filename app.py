from __future__ import print_function

import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from env import Environment
from game import CFRRL_Game
from model import Network
from config import get_config

# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'data'


# Load flags and configuration
FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('ckpt', '', 'Apply a specific checkpoint')
tf.compat.v1.flags.DEFINE_boolean('eval_delay', False, 'Evaluate delay or not')

# Load config and initialize TensorFlow environment, game, and network
config = get_config(FLAGS) or FLAGS
env = Environment(config, is_training=False)
game = CFRRL_Game(config, env)
network = Network(config, game.state_dims, game.action_dim, game.max_moves)
network.restore_ckpt(FLAGS.ckpt)

# Setup for TensorFlow CPU usage
tf.config.experimental.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('INFO')

def sim(config, network, game):
    results = []

    for tm_idx in game.tm_indexes:
        state = game.get_state(tm_idx)
        if config.method == 'actor_critic':
            policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif config.method == 'pure_policy':
            policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        actions = policy.argsort()[-game.max_moves:]

        # Display connections
        for idx in actions:
            if idx >= 132:
                continue
            src = idx // (12 - 1)
            dst = idx % (12 - 1)
            if dst >= src:
                dst += 1
            print(f"{idx}: node {src} -> node {dst}")

        # Evaluate solution and collect result
        solution_rs = game.evaluate(tm_idx, actions, eval_delay=FLAGS.eval_delay)
        results.append(solution_rs)
    
    return results
def convert_to_basic_types(data):
    if isinstance(data, list):
        return [convert_to_basic_types(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_basic_types(item) for item in data)
    elif isinstance(data, dict):
        return {key: convert_to_basic_types(value) for key, value in data.items()}
    elif isinstance(data, (np.int64, np.float64)):  # Kiểm tra nếu dữ liệu là kiểu int64 hoặc float64
        return int(data)
    else:
        return data
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file as 'AbileneTM3' in the data folder
        file_path = os.path.join(UPLOAD_FOLDER, 'AbileneTM4')
        file.save(file_path)
        
        # Load the traffic matrix data from the file
        # (Assuming that the file is formatted in a compatible way)
        tf.config.experimental.set_visible_devices([], 'GPU')
        tf.get_logger().setLevel('INFO')

        config = get_config(FLAGS) or FLAGS
        env = Environment(config, is_training=False)
        game = CFRRL_Game(config, env)
        network = Network(config, game.state_dims, game.action_dim, game.max_moves)
        print("game action dim", game.action_dim)
        step = network.restore_ckpt(FLAGS.ckpt)
        if config.method == 'actor_critic':
            learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
        elif config.method == 'pure_policy':
            learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
        print('\nstep %d, learning rate: %f\n'% (step, learning_rate))
        # Run the simulation and get the results
        solution_results = sim(config, network, game)
        converted_solution_results = convert_to_basic_types(solution_results)
        print(converted_solution_results)
        
        # Return the solution results as JSON
        return jsonify({'solution_rs': converted_solution_results})

# Flask app entry point
if __name__ == '__main__':
    # Set host='0.0.0.0' to make the server externally visible if needed
    app.run(host='0.0.0.0', port=5000)
