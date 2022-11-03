# Variables
mflogFile=.cache/mflog.txt
da_cache=.cache/output/for*.pickle
rt_cache=.cache/output/realtime_data.csv

# Target
rt_run: $(rt_cache) $(da_cache)

$(da_cache):
	python src/dayahead_alg.py
$(rt_cache): $(da_cache)
	python src/realtime_alg.py
