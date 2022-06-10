from dayahead_alg import *
from realtime_alg import *
dt = datetime.strptime

if __name__ == '__main__':
    VEC_INP = open(r"data/test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
    PREVIOUS_DAYS = 30
    CASES = {{"robust_par": 1, "mode_forecast": "Clustering", "date": dt("2022-06-01"), "previous_days": 30},
             {"robust_par": 1, "mode_forecast": "ARIMA", "date": dt("2022-06-01"), "previous_days": 30},
             {"robust_par": 1, "mode_forecast": "MarkovChain", "date": dt("2022-06-01"), "previous_days": 30}}
    obj_list = []
    dev_list = []
    out_sample_res1_df = pd.DataFrame()
    out_sample_res2_DF = pd.DataFrame()
    for CASE in CASES:
        obj = day_ahead_alg(*CASE)
        obj_list.append(obj)
        dev = 0
        for t in range(10):
            DATE = CASE["date"] + timedelta(days=1) + timedelta(minutes=10 * t)
            _, rt_res = interface_control_digriflex(vec_inp=VEC_INP, date=DATE)
            dev += abs(rt_res)
        dev_list.append(dev)
    # %%
    plt.style.use('default')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=20)

    fig, ax = plt.subplots(1, 1)
    plt.bar(obj_list)
    ax.set_ylabel('Objective (\$)', fontsize=20)
    fig.savefig(r".cache/figures/objectives.pdf", bbox_inches='tight')
    fig, ax = plt.subplots(1, 1)
    plt.bar(dev_list)
    ax.set_ylabel('Deviation amount (pu)', fontsize=20)
    fig.savefig('.cache/figures/res.pdf', bbox_inches='tight')
    # fig, ax = plt.subplots(1, 1)
    # montec_res2_DF.plot.box(ax=ax, whis=1, showfliers=False, rot=90)
    # ax.set_ylabel('Deviation probability (pu)', fontsize=20)
    # fig.savefig('./Figures/res2.png', bbox_inches='tight')
