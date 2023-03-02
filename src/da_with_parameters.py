from dayahead_alg import dayahead_alg
from datetime import datetime
import logging


# Global variables
# ----------------------------------------------------------------------------------------------------------------------
dt = datetime.strptime
log = logging.getLogger(__name__)


if __name__ == '__main__':
    MODE_FORECAST = "BayesBoot"
    NUM_DAYS = 30
    MODE_OPT = "Robust"
    ROBUST_PAR = 0.8
    DATE = datetime.now().strftime("%Y-%m-%d")
    _, flag = dayahead_alg(robust_par=ROBUST_PAR, mode_forecast=MODE_FORECAST, date=dt(DATE, "%Y-%m-%d"),
                           previous_days=NUM_DAYS)
    log.info("Day_ahead optimization finished: {}".format(flag))
