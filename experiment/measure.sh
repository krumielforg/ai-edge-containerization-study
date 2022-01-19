#!/bin/bash

PID=$1
LOGFILE=$2

echo $(date -u "+%D %T.%3N") $(sudo top -n 1 -b -p $PID | tail -2 | head -1) >> $LOGFILE
echo $(date -u "+%D %T.%3N") $(sudo pidstat -p $PID -ruh | head -3) >> $LOGFILE
while $(ps -p $PID >/dev/null)
do
    # echo $(date -u "+%D %T.%3N") $(sudo ps -p $PID -o pid,%cpu,%mem | tail -1) $(sudo pmap $PID | tail -n 1 | awk '/[0-9]K/{print $2}')
    echo $(date -u "+%D %T.%3N") $(sudo top -n 1 -b -p $PID | tail -1) >> $LOGFILE
    echo $(date -u "+%D %T.%3N") $(sudo pidstat -p $PID -ruh | tail -n+4) >> $LOGFILE
    # sleep 1 # seconds
    sleep 0.1
    # sleep 0.001 # milliseconds (sampling rate)
done
