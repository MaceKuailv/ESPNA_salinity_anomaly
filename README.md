This repo stores the code to reproduce results and plots from "Generation and propagation of Eastern Subpolar North Atlantic salinity anomalies". To reproduce follow the following steps (change paths as appropriate): 

1. Get the data. ECCO v4r4 product is openly available at: https://ecco-group.org/products-ECCO-V4r4.htm
2. Set up the environment. The main machinary behind the analysis is seaduck. Most of the analysis is conducted using seaduck 1.0.2. Newer version of seaduck is expected to continuously support the code.
3. Run scripts and notebooks in the budget_prep folder to get the Eulerian salinity budget. It is a bit unfortunate, but at this point, you need an older version of xgcm (e.g., 0.6.1) to create budget in ECCO. (The xgcm project was abandoned after some breaking changes were introduced.)
4. Run lag_budget. The shell scripts are for parallizing the operation.
5. Run notebooks in "plotting" to get the plots.

If you have run into any problems or have any question, please contact Wenrui Jiang: wjiang33@jhu.edu. 