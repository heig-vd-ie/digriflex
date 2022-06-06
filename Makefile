# Variables
mflogFile=.cache/mflog.txt
da_cache=.cache/output/for*.pickle
rt_cache=.cache/rolle/realtime_data.csv

# Target
rt_run: $(rt_cache) $(da_cache)

$(da_cache):
	python src/da_optimize.py
$(rt_cache): $(da_cache)
	python src/rt_optimize.py
