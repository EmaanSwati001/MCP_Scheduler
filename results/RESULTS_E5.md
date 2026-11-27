# E5: Heterogeneous server results (N2)

Mixed topology: same 15 Venus clusters, 1,080 GPUs total. Each cluster: floor(S/2) eight-GPU servers split into two four-GPU servers; remaining servers keep 8 GPUs.

## venus eta=15

### N2-A: Homogeneous topology

| run | avg_wait | p99_wait | max_wait | starved_3x | wait_8gpu | util | lb_index | unalloc |
|---|---|---|---|---|---|---|---|---|
| fcfs | 33590 | 86647 | 90844 | 3597 | 24035 | 0.1931 | 0.01114 | 0.1134 |
| consolidate | 25258 | 69589 | 94464 | 2923 | 28881 | 0.1925 | 0.02358 | 0.1079 |
| fgd | 23581 | 68045 | 86586 | 2908 | 29680 | 0.1924 | 0.01138 | 0.1036 |
| mast | 13374 | 109493 | 126839 | 745 | 55285 | 0.1877 | 0.01220 | 0.2200 |
| mcg | 28310 | 83625 | 106223 | 3419 | 33100 | 0.1922 | 0.00577 | 0.1539 |

### N2-A: Mixed topology

| run | avg_wait | p99_wait | max_wait | starved_3x | wait_8gpu | util | lb_index | unalloc |
|---|---|---|---|---|---|---|---|---|
| fcfs | 91497 | 189376 | 196039 | 5559 | 76257 | 0.1860 | 0.01408 | 0.3724 |
| consolidate | 14674 | 143559 | 182116 | 922 | 71289 | 0.1863 | 0.01119 | 0.3089 |
| fgd | 14584 | 143780 | 183140 | 924 | 71923 | 0.1862 | 0.01091 | 0.3104 |
| mast | 20792 | 201747 | 239835 | 1030 | 111594 | 0.1811 | 0.00730 | 0.4149 |
| mcg | 71843 | 149325 | 187373 | 4406 | 70636 | 0.1864 | 0.00648 | 0.3457 |

### N2-B: F vs F_C on mixed topology

| run | avg_wait | p99_wait | max_wait | starved_3x | wait_8gpu | util | lb_index | unalloc |
|---|---|---|---|---|---|---|---|---|
| mcg | 71843 | 149325 | 187373 | 4406 | 70636 | 0.1864 | 0.00648 | 0.3457 |
| mcg_fc | 77675 | 159209 | 202358 | 4469 | 74890 | 0.1862 | 0.00904 | 0.3670 |
| mcg_fc_norm | 77568 | 159114 | 201823 | 4469 | 74882 | 0.1862 | 0.00906 | 0.3668 |


**Key finding:** Mixed topology raises MCG avg wait ×2.5 (71,843 s vs 28,310 s). MCG is still best on mixed (71,843 s vs FGD 14,584 s).

**F_C result:** negative – F_C: 77,675 s, F_C (renorm): 77,568 s vs original: 71,843 s.

## venus eta=10

### N2-A: Homogeneous topology

| run | avg_wait | p99_wait | max_wait | starved_3x | wait_8gpu | util | lb_index | unalloc |
|---|---|---|---|---|---|---|---|---|
| fcfs | 17891 | 54678 | 57192 | 2426 | 11182 | 0.1913 | 0.01135 | 0.1602 |
| consolidate | 14610 | 44175 | 63693 | 2363 | 12858 | 0.1913 | 0.01361 | 0.1628 |
| fgd | 14949 | 46025 | 65447 | 2383 | 13817 | 0.1913 | 0.01445 | 0.1677 |
| mast | 8718 | 73998 | 90684 | 584 | 32749 | 0.1894 | 0.00948 | 0.2377 |
| mcg | 10747 | 51891 | 67038 | 349 | 17221 | 0.1913 | 0.00594 | 0.1820 |

### N2-A: Mixed topology

| run | avg_wait | p99_wait | max_wait | starved_3x | wait_8gpu | util | lb_index | unalloc |
|---|---|---|---|---|---|---|---|---|
| fcfs | 71018 | 151141 | 157295 | 4936 | 59464 | 0.1851 | 0.01294 | 0.3826 |
| consolidate | 10547 | 112344 | 148797 | 873 | 55955 | 0.1859 | 0.01103 | 0.3335 |
| fgd | 10587 | 110829 | 148989 | 889 | 55675 | 0.1859 | 0.01171 | 0.3341 |
| mast | 18865 | 182301 | 218798 | 999 | 101412 | 0.1794 | 0.00985 | 0.4477 |
| mcg | 53872 | 122327 | 149652 | 4103 | 53910 | 0.1860 | 0.00755 | 0.3582 |

### N2-B: F vs F_C on mixed topology

| run | avg_wait | p99_wait | max_wait | starved_3x | wait_8gpu | util | lb_index | unalloc |
|---|---|---|---|---|---|---|---|---|
| mcg | 53872 | 122327 | 149652 | 4103 | 53910 | 0.1860 | 0.00755 | 0.3582 |
| mcg_fc | 64499 | 131010 | 168456 | 4195 | 61328 | 0.1850 | 0.00669 | 0.3840 |
| mcg_fc_norm | 59686 | 129101 | 160928 | 4351 | 61416 | 0.1850 | 0.00657 | 0.3800 |


**Key finding:** Mixed topology raises MCG avg wait ×5.0 (53,872 s vs 10,747 s). MCG is still best on mixed (53,872 s vs FGD 10,587 s).

**F_C result:** negative – F_C: 64,499 s, F_C (renorm): 59,686 s vs original: 53,872 s.
