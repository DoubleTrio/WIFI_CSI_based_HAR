ssh root@192.168.1.1 -C "./receive_data.sh $1" &
ssh root@192.168.2.1 -C "./send_csi_data.sh $2" &
python3 run_visualization_server.py