input=$1
cnt=1
while IFS= read -r line
do
    echo "$line"
    ./solver $line 46 37 30 11 ./schedule_46_37_30_11_fixY/
    ./solver $line 60 37 30 11 ./schedule_60_37_30_11_fixY/
    ./solver $line 65 57 28 16 ./schedule_65_57_28_16_fixY/
    echo $cnt
    ((cnt++))

done < "$input"

