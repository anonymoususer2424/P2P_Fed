#!/bin/bash

tc qdisc replace dev eth0 root handle 1: htb default 14
tc class add dev eth0 parent 1: classid 1:1 htb rate 1000mbit
tc class add dev eth0 parent 1:1 classid 1:11 htb rate 100mbit ceil 100mbit
tc qdisc add dev eth0 parent 1:11 handle 11: fq_codel
tc filter add dev eth0 protocol ip prio 1 parent 1: u32 match ip src 0.0.0.0/0 classid 1:11


ip link add name ifb0 type ifb
ip link set dev ifb0 up
tc qdisc add dev eth0 handle ffff: ingress
tc filter add dev eth0 parent ffff: protocol all matchall action mirred egress redirect dev ifb0

tc qdisc replace dev ifb0 root handle 1: htb default 14
tc class add dev ifb0 parent 1: classid 1:1 htb rate 1000mbit
tc class add dev ifb0 parent 1:1 classid 1:11 htb rate 100mbit ceil 100mbit
tc qdisc add dev ifb0 parent 1:11 handle 11: fq_codel
tc filter add dev ifb0 protocol ip prio 1 parent 1: u32 match ip dst 0.0.0.0/0 classid 1:11