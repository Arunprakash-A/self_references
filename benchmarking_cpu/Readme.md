# Benchmark the inference performance of small language models (size 1B to 4B) on CPU, GPU and OpenVINO
* Model: google/gemma-2b
* Input tokens: 13
* Max new tokens: 128
* Runs: 5

**Intel(R) Xeon(R) Gold 6430**

**CPU**
* Dtype: Float32 <br>
* Num of CPU Cores: 64 <br>

```
TTFT (s): [0.2275, 0.2016, 0.2567, 0.2135, 0.3343]
Latency total (s): [10.8356, 9.3398, 9.2782, 9.3191, 8.9173]
Throughput (tok/s): [11.81, 13.7, 13.8, 13.74, 14.35]
Generated tokens: [128, 128, 128, 128, 128]
```

**With OpenVINO**
* Weight Compression: INT8 
* Model: google/gemma-2b
* Device: CPU
* OV config: {'PERFORMANCE_HINT': 'LATENCY'}
```
TTFT approx (s): [0.0607, 0.0738, 0.0456, 0.0458, 0.0575]
Latency total (s): [5.9138, 5.5445, 5.7253, 5.4596, 5.1725]
Throughput (tok/s): [21.64, 23.09, 22.36, 23.44, 24.75]
Generated tokens: [128, 128, 128, 128, 128]
```

**L40 GPU**
* Dtype: bf16
```
TTFT (s): [0.0288, 0.0282, 0.0263, 0.0263, 0.0262]
Latency total (s): [3.4132, 3.0791, 3.0515, 3.0573, 3.0502]
Throughput (tok/s): [37.5, 41.57, 41.95, 41.87, 41.96]
Generated tokens: [128, 128, 128, 128, 128]
```
**In Collab: Intel(R) Xeon(R) **
**CPU**
Num of CPU cores: 2
```
TTFT (s): [2.2701, 3.1659, 2.2591, 2.3595, 2.7969]
Latency total (s): [118.4143, 119.3708, 118.3294, 117.6054, 117.1872]
Throughput (tok/s): [1.08, 1.07, 1.08, 1.09, 1.09]
Generated tokens: [128, 128, 128, 128, 128]
```
**With OpenVINO**

* Device: CPU
* OV config: {'PERFORMANCE_HINT': 'LATENCY'}
```
TTFT approx (s): [1.6748, 1.69, 2.0288, 1.6856, 2.3686]
Latency total (s): [57.4901, 55.4858, 56.2848, 55.6641, 56.9789]
Throughput (tok/s): [2.23, 2.31, 2.27, 2.3, 2.25]
Generated tokens: [128, 128, 128, 128, 128]
```
**T4 GPU**
* Dtype: Float32
```
TTFT (s): [0.0797, 0.0596, 0.06, 0.0605, 0.0606]
Latency total (s): [5.7272, 5.6574, 5.7158, 5.6702, 5.714]
Throughput (tok/s): [22.35, 22.63, 22.39, 22.57, 22.4]
Generated tokens: [128, 128, 128, 128, 128]
```
**T4 GPU**
* Dtype: bf16
```
TTFT (s): [0.1293, 0.1688, 0.1347, 0.1322, 0.1335]
Latency total (s): [5.994, 3.9013, 3.7612, 4.7523, 3.6536]
Throughput (tok/s): [21.35, 32.81, 34.03, 26.93, 35.03]
Generated tokens: [128, 128, 128, 128, 128]
```
