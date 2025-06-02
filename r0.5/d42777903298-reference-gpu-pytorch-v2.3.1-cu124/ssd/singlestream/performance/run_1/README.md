*Check [MLC MLPerf docs](https://docs.mlcommons.org/automotive) for more details.*

## Host platform

* OS version: Linux-6.8.0-1030-gcp-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0]
* MLC version: unknown

## MLC Run Command

See [MLC installation guide](https://docs.mlcommons.org/mlcflow/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo anandhu-eng@mlperf-automations --checkout=43d757b7165d6611408380c5984d7d3108e2818f


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload anandhu-eng@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo anandhu-eng@mlperf-automations
mlc pull repo anandhu-eng@mlperf-automations
mlc rm cache -f

```