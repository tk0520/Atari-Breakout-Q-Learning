#!/bin/bash

read -p "Enter PID: " pid;

sysctl -w vm.min_free_kbytes=262144
sysctl -w vm.swappiness=90

renice -n -10 -p $pid
ionice -c 2 -n 0 -p $pid
taskset -cp 0-7 $pid
