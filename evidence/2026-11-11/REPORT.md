# 实验自动汇总

每行报告均值与样本标准差（ddof=1）。MSE 使用原始目标的平方单位。

|实验组|任务|模型|种子数|MSE|R²|F1|CF|
|---|---|---|---|---|---|---|---|
|ablations|mixed30|legacy|3|1.7671 ± 0.2|0.976 ± 0.00433|0.28571 ± 0|0.6 ± 0|
|ablations|mixed30|no_fs|3|1.6517 ± 0.215|0.97755 ± 0.00418|0.28571 ± 0|0.6 ± 0|
|ablations|mixed30|no_l1|3|1.5737 ± 0.244|0.97857 ± 0.00488|0.28571 ± 0|0.6 ± 0|
|ablations|mixed30|no_l2|3|1.5746 ± 0.214|0.97856 ± 0.0045|0.28571 ± 0|0.6 ± 0|
|ablations|mixed30|no_reg|3|1.5729 ± 0.133|0.97864 ± 0.00335|0.28571 ± 0|0.6 ± 0|
|ablations|mixed30|no_sd|3|1.5206 ± 0.141|0.97934 ± 0.00337|0.28571 ± 0|0.6 ± 0|
|ablations|mixed30|nplam|3|1.541 ± 0.258|0.97899 ± 0.00508|0.28571 ± 0|0.6 ± 0|
|ablations|mixed30|train_hidden|3|1.4576 ± 0.0214|0.98029 ± 0.0014|0.28571 ± 0|0.6 ± 0|
|highdim|sparse300|lasso|3|15.632 ± 1.5|0.77889 ± 0.0141|0.75 ± 0|0.4 ± 6.8e-17|
|highdim|sparse300|mlp|3|43.034 ± 1.22|0.38955 ± 0.0414|N/A|N/A|
|highdim|sparse300|nam|3|8.7094 ± 1.4|0.87573 ± 0.0269|N/A|N/A|
|highdim|sparse300|nplam|3|37.139 ± 4.88|0.47072 ± 0.0987|0.032859 ± 6.23e-05|0.6 ± 0|
|real|beijing|lasso|3|699.35 ± 1.39e-13|0.90377 ± 1.36e-16|N/A|N/A|
|real|beijing|mlp|3|632.34 ± 37.9|0.91299 ± 0.00522|N/A|N/A|
|real|beijing|nam|3|789.84 ± 78.1|0.89132 ± 0.0107|N/A|N/A|
|real|beijing|nplam|3|794.84 ± 57.2|0.89063 ± 0.00787|N/A|N/A|
|real|california|lasso|3|4.6851e+09 ± 3.49e+07|0.64001 ± 0.0091|N/A|N/A|
|real|california|mlp|3|3.3213e+09 ± 5.04e+07|0.74478 ± 0.00837|N/A|N/A|
|real|california|nam|3|3.8492e+09 ± 1.03e+08|0.70421 ± 0.0116|N/A|N/A|
|real|california|nplam|3|3.9822e+09 ± 7.71e+07|0.69398 ± 0.0112|N/A|N/A|
|real|superconductivity|lasso|3|318.91 ± 20.6|0.72782 ± 0.0178|N/A|N/A|
|real|superconductivity|mlp|3|198.26 ± 16.6|0.83071 ± 0.0156|N/A|N/A|
|real|superconductivity|nam|3|235.54 ± 21.6|0.79903 ± 0.0177|N/A|N/A|
|real|superconductivity|nplam|3|288.08 ± 5.29|0.75411 ± 0.00634|N/A|N/A|
|synthetic|dense30|lasso|10|83.862 ± 8.62|0.76991 ± 0.0326|0.73925 ± 0.0189|0.49333 ± 0.0141|
|synthetic|dense30|mlp|10|82.73 ± 8.72|0.77192 ± 0.0384|N/A|N/A|
|synthetic|dense30|nam|10|2.8564 ± 0.299|0.99216 ± 0.00116|N/A|N/A|
|synthetic|dense30|nplam|10|11.103 ± 2.83|0.96942 ± 0.00853|1 ± 0|0.5 ± 0|
|synthetic|linear30|lasso|10|1.0226 ± 0.0695|0.98983 ± 0.000813|1 ± 0|1 ± 0|
|synthetic|linear30|mlp|10|1.9965 ± 0.234|0.9801 ± 0.00278|N/A|N/A|
|synthetic|linear30|nam|10|1.1528 ± 0.0779|0.98851 ± 0.00122|N/A|N/A|
|synthetic|linear30|nplam|10|1.3428 ± 0.188|0.98662 ± 0.00213|0.62795 ± 0.314|0.48 ± 0.253|
|synthetic|mixed30|lasso|10|15.042 ± 1.43|0.80018 ± 0.0162|0.77778 ± 0.0586|0.4 ± 0|
|synthetic|mixed30|mlp|10|16.984 ± 2.28|0.77417 ± 0.0298|N/A|N/A|
|synthetic|mixed30|nam|10|1.3447 ± 0.105|0.98208 ± 0.00176|N/A|N/A|
|synthetic|mixed30|nplam|10|1.5534 ± 0.206|0.9793 ± 0.00312|0.45325 ± 0.275|0.74 ± 0.0966|
|synthetic|nonlinear30|lasso|10|28.07 ± 2.82|0.19106 ± 0.0154|0.32857 ± 0.0151|0 ± 0|
|synthetic|nonlinear30|mlp|10|16.432 ± 1.37|0.52118 ± 0.0712|N/A|N/A|
|synthetic|nonlinear30|nam|10|1.8346 ± 0.405|0.94682 ± 0.0127|N/A|N/A|
|synthetic|nonlinear30|nplam|10|1.8116 ± 0.391|0.94739 ± 0.0121|0.51575 ± 0.312|1 ± 0|
