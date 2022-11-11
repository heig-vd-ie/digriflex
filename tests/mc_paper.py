from dayahead_alg import dayahead_alg
from datetime import datetime, timedelta
from realtime_alg import interface_control_digriflex
import coloredlogs
import logging
import matplotlib.pyplot as plt
import random


# Global variables
# ----------------------------------------------------------------------------------------------------------------------
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
dt = datetime.strptime

if __name__ == '__main__':
    VEC_INP = open(r"data/test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
    number_prob = 1
    CASES = [
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-01-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-02-27", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-03-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-04-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-05-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-06-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-07-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-08-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-09-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-10-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-11-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "NoClustering", "date": dt("2021-12-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-01-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-02-27", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-03-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-04-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-05-30", "%Y-%m-%d"), "previous_days": 30},
        {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-06-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-07-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-08-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-09-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-10-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-11-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2021-12-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-01-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-02-27", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-03-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-04-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-05-30", "%Y-%m-%d"), "previous_days": 30},
        {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-06-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-07-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-08-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-09-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-10-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-11-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2021-12-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-01-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-02-27", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-03-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-04-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-05-30", "%Y-%m-%d"), "previous_days": 30},
        {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-06-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-07-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-08-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-09-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-10-30", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-11-29", "%Y-%m-%d"), "previous_days": 30},
        # {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2021-12-30", "%Y-%m-%d"), "previous_days": 30},
    ]
    obj_list = []
    dev_list = []
    for CASE in CASES:
        log.info("Case of " + str(CASE["date"]))
        obj = dayahead_alg(**CASE)
        obj_list.append(obj)
        dev = 0
        times = range(1, 143) if number_prob >= 144 else [random.randint(1, 143) for _ in range(number_prob)]
        for t in times:
            date = CASE["date"] + timedelta(days=1) + timedelta(minutes=10 * t)
            _, s1 = interface_control_digriflex(vec_inp=VEC_INP, date=date)
            dev += s1
        log.info("Deviation of this month is " + str(dev * 100 / number_prob) + "%.")
        dev_list.append(dev * 100 / number_prob)
    plt.style.use('default')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=20)
    fig, ax = plt.subplots(1, 1)
    plt.plot(obj_list)
    ax.set_ylabel('Objective (\$)', fontsize=20)
    fig.savefig(r".cache/figures/objectives.pdf", bbox_inches='tight')
    fig, ax = plt.subplots(1, 1)
    plt.plot(dev_list)
    ax.set_ylabel('Deviation (\%)', fontsize=20)
    fig.savefig(r".cache/figures/deviations.pdf", bbox_inches='tight')
    with open(".cache/figures/objectives.txt", 'w') as f:
        for item in obj_list:
            f.write(str(item) + "\n")
    with open(".cache/figures/deviations.txt", 'w') as f:
        for item in dev_list:
            f.write(str(item) + "\n")
