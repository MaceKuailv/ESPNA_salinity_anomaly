for i in $(seq 0 9)
do
    # python Mfresh_run.py $i &
    python fresh_run_mean.py $i &
done