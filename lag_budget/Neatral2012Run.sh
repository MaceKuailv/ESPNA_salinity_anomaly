for i in $(seq 0 8)
do
    # python neutral_2012_time_dependent.py $i &
    python neutral_2012_mean.py $i &
done