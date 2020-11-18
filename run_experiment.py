
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import webbrowser

import numpy as np
import torch.optim as optim

import utils


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)

plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
np.set_printoptions(suppress=True)


def main(_args):
    agent_type = _args.agent
    policy_type = _args.policy
    results_path = _args.results_path

    results = []
    best_agent, best_eval_score = None, float('-inf')

    if agent_type == "reinforce":
        from scripts.agent_reinforce import REINFORCE, PolicyNetReinforce
        AGENT = REINFORCE
        policy_net = PolicyNetReinforce
    elif agent_type == "vpg":
        from scripts.agent_vpg import VPG, PolicyNetVPG
        AGENT = VPG
        policy_net = PolicyNetVPG
    else:
        raise NotImplementedError("Other Agent types are not supported yet")

    utils.create_directory(results_path)

    for seed in SEEDS:
        environment_settings = {
            'env_name': 'CartPole-v1',
            'gamma': 1.00,
            'max_minutes': 10,
            'max_episodes': 10000,
            'goal_mean_100_reward': 475
        }

        policy_model_fn = lambda nS, nA: policy_net(nS, nA, hidden_dims=(128, 64))
        policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        policy_optimizer_lr = 0.0005

        env_name, gamma, max_minutes, \
        max_episodes, goal_mean_100_reward = environment_settings.values()

        agent = AGENT(policy_model_fn, policy_optimizer_fn, policy_optimizer_lr)

        make_env_fn, make_env_kargs = utils.get_make_env_fn(env_name=env_name)
        # make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name, unwrapped=True)
        # make_env_fn, make_env_kargs = get_make_env_fn(
        #     env_name=env_name, addon_wrappers=[MCCartPole,])
        result, final_eval_score, training_time, wallclock_time = agent.train(
            make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
        results.append(result)
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent = agent
    results = np.array(results)
    _ = BEEP()

    # Agent Progression
    html_data, title = best_agent.demo_progression()
    fig_path = utils.save_html(data=html_data, path=os.path.join(results_path, f"{title}.html"))
    webbrowser.open_new_tab(fig_path)

    # Best Agent
    html_data, title = best_agent.demo_last()
    fig_path = utils.save_html(data=html_data, path=os.path.join(results_path, f"{title}.html"))
    webbrowser.open_new_tab(fig_path)

    # Extracting statistics
    agent_max_t, agent_max_r, agent_max_s, \
    agent_max_sec, agent_max_rt = np.max(results, axis=0).T

    agent_min_t, agent_min_r, agent_min_s, \
    agent_min_sec, agent_min_rt = np.min(results, axis=0).T

    agent_mean_t, agent_mean_r, agent_mean_s, \
    agent_mean_sec, agent_mean_rt = np.mean(results, axis=0).T

    agent_x = np.arange(len(agent_mean_s))

    # Plot Statistics
    fig, axs = plt.subplots(5, 1, figsize=(20, 30), sharey=False, sharex=True)

    axs[0].plot(agent_max_r, 'b', linewidth=1)
    axs[0].plot(agent_min_r, 'b', linewidth=1)
    axs[0].plot(agent_mean_r, 'b', label=agent_type.upper(), linewidth=2)
    axs[0].fill_between(agent_x, agent_min_r, agent_max_r, facecolor='b', alpha=0.3)

    axs[1].plot(agent_max_s, 'b', linewidth=1)
    axs[1].plot(agent_min_s, 'b', linewidth=1)
    axs[1].plot(agent_mean_s, 'b', label=agent_type.upper(), linewidth=2)
    axs[1].fill_between(agent_x, agent_min_s, agent_max_s, facecolor='b', alpha=0.3)

    axs[2].plot(agent_max_t, 'b', linewidth=1)
    axs[2].plot(agent_min_t, 'b', linewidth=1)
    axs[2].plot(agent_mean_t, 'b', label=agent_type.upper(), linewidth=2)
    axs[2].fill_between(agent_x, agent_min_t, agent_max_t, facecolor='b', alpha=0.3)

    axs[3].plot(agent_max_sec, 'b', linewidth=1)
    axs[3].plot(agent_min_sec, 'b', linewidth=1)
    axs[3].plot(agent_mean_sec, 'b', label=agent_type.upper(), linewidth=2)
    axs[3].fill_between(agent_x, agent_min_sec, agent_max_sec, facecolor='b', alpha=0.3)

    axs[4].plot(agent_max_rt, 'b', linewidth=1)
    axs[4].plot(agent_min_rt, 'b', linewidth=1)
    axs[4].plot(agent_mean_rt, 'b', label=agent_type.upper(), linewidth=2)
    axs[4].fill_between(agent_x, agent_min_rt, agent_max_rt, facecolor='b', alpha=0.3)

    # ALL
    axs[0].set_title('Moving Avg Reward (Training)')
    axs[1].set_title('Moving Avg Reward (Evaluation)')
    axs[2].set_title('Total Steps')
    axs[3].set_title('Training Time')
    axs[4].set_title('Wall-clock Time')
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')

    fig.savefig(os.path.join(results_path, f"{agent_type.upper()}_Statistics.png"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="reinforce",
                        help="agent type")

    parser.add_argument("--results_path", type=str,
                        default="results/run_{}".format(utils.get_date_time_now()),
                        help="path to save the experiment results")

    args = parser.parse_args()
    main(args)
