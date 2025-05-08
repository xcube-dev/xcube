import pstats

stats = pstats.Stats("profile_output.prof")
stats.sort_stats("cumulative").print_stats("reproject")
