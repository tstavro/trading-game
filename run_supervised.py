'''
    Supervised model
'''

from Util import RandomWalkEnv
import tensorflow as tf


tf.compat.v1.disable_eager_execution()

model = tf.keras.models.load_model('./models/supervised.keras')


def play_game(model_path=''):
    # model.load()
    observation = env.reset()
    done = False
    cnt = 0  # number of moves in an episode
    while not done:
        cnt += 1
        action = model.predict(observation.reshape(1,-1))
        if action > 0.5:
            action = 0
        else:
            action = 1
        observation, reward, done, _ = env.step(action)
    return cnt

for i in range(18, 30):
    env = RandomWalkEnv(size=10000, random_seed=i, equity=1)
    play_game('')
    env.disp_dataset('Supervised')
    # env.disp_equity('Supervised - accuracy:'+ str(round(env.accuracy, 4)), saveFig=True)
    env.disp_equity('Supervised - accuracy:'+ str(round(env.accuracy, 4)))
    print('accuracy:', env.accuracy)



