
import tempfile
import gym
from gym import wrappers
import numpy as np
import json
import io
import base64
import os
import subprocess
from pathlib import Path
from datetime import datetime
import pickle

import torch


def get_make_env_fn(**kargs):
    def make_env_fn(env_name, seed=None, render=None, record=False,
                    unwrapped=False, monitor_mode=None,
                    inner_wrappers=None, outer_wrappers=None):
        mdir = tempfile.mkdtemp()
        env = None
        if render:
            try:
                env = gym.make(env_name, render=render)
            except:
                pass
        if env is None:
            env = gym.make(env_name)
        if seed is not None: env.seed(seed)
        env = env.unwrapped if unwrapped else env
        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)
        env = wrappers.Monitor(
            env, mdir, force=True,
            mode=monitor_mode,
            video_callable=lambda e_idx: record) if monitor_mode else env
        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)
        return env
    return make_env_fn, kargs


def get_videos_html(env_videos, title, max_n_videos=5):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return

    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1, ]
    videos = videos[idxs, ...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        video = io.open(video_path, 'r+b').read()
        encoded = base64.b64encode(video)

        with open(meta_path) as data_file:
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <video width="960" height="540" controls>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>"""
        strm += html_tag.format('Episode ' + str(meta['episode_id']), encoded.decode('ascii'))
    return strm


def get_gif_html(env_videos, title, subtitle_eps=None, max_n_videos=4):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return

    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1, ]
    videos = videos[idxs, ...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        basename = os.path.splitext(video_path)[0]
        gif_path = basename + '.gif'
        if not os.path.exists(gif_path):
            ps = subprocess.Popen(
                ('ffmpeg',
                 '-i', video_path,
                 '-r', '7',
                 '-f', 'image2pipe',
                 '-vcodec', 'ppm',
                 '-crf', '20',
                 '-vf', 'scale=512:-1',
                 '-'),
                stdout=subprocess.PIPE)
            output = subprocess.check_output(
                ('convert',
                 '-coalesce',
                 '-delay', '7',
                 '-loop', '0',
                 '-fuzz', '2%',
                 '+dither',
                 '-deconstruct',
                 '-layers', 'Optimize',
                 '-', gif_path),
                stdin=ps.stdout)
            ps.wait()

        gif = io.open(gif_path, 'r+b').read()
        encoded = base64.b64encode(gif)

        with open(meta_path) as data_file:
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <img src="data:image/gif;base64,{1}" />"""
        prefix = 'Trial ' if subtitle_eps is None else 'Episode '
        sufix = str(meta['episode_id'] if subtitle_eps is None \
                        else subtitle_eps[meta['episode_id']])
        strm += html_tag.format(prefix + sufix, encoded.decode('ascii'))
    return strm


def save_html(data, path):
    f = open(path, 'wt')
    f.write(data)
    f.close()
    print(f"HTML file saved successfully to : {path}")
    return path


def save_checkpoint(checkpoint_dir, episode_idx, model):
    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, 'model.{}.tar'.format(episode_idx)))


def get_default_variable_dict():
    variable_dict = {
        "LEAVE_PRINT_EVERY_N_SECS": 30,
        "ERASE_LINE": '\x1b[2K'
    }
    return variable_dict


def get_date_time_now():
    return str(datetime.now().replace(microsecond=0)).replace(':', '_').replace(' ', '_')


def create_directory(directory: str):

    if not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        print("Directory ", directory, " Created ")
    else:
        print("Directory ", directory, " already exists")


def save_data(data, path: str):
    with open(path, 'wb') as pickle_data:
        pickle.dump(data, pickle_data)
        print(f"INFO: Data saved successfully to {path}")
