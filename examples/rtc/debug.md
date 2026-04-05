# 1.
inference_latency 中保存的其实是以前所有推理过程中的最大延迟

这是计算的当前推理的真实延迟
new_latency = time.perf_counter() - current_time
new_delay = math.ceil(new_latency / time_per_chunk)


















