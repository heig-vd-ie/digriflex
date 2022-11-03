from dayahead_alg import dayahead_alg
from datetime import datetime
import matplotlib.pyplot as plt

dt = datetime.strptime

if __name__ == '__main__':
    VEC_INP = open(r"data/test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
    PREVIOUS_DAYS = 30
    CASES = [
        {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-02-15", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-03-15", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-03-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-04-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-05-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-06-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-07-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-08-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-09-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-10-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-11-01", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-12-01", "%Y-%m-%d"), "previous_days": 30},
    ]
    obj_list = []
    for CASE in CASES:
        obj = dayahead_alg(**CASE)
        obj_list.append(obj)
    plt.style.use('default')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=20)
    fig, ax = plt.subplots(1, 1)
    plt.plot(obj_list)
    ax.set_ylabel('Objective (\$)', fontsize=20)
    fig.savefig(r".cache/figures/objectives.pdf", bbox_inches='tight')
    fig, ax = plt.subplots(1, 1)
