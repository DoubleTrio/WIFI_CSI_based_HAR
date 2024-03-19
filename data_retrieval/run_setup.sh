# Example usage: ./run_setup.sh <IP_ADDRESS> <PACKETS_PER_SECOND>

ssh root@192.168.1.1 -C "./receive_data.sh $1" &
ssh root@192.168.2.1 -C "./send_csi_data.sh $2" &
python3 run_visualization_server.py &

function no_ctrlc()
{
    MIRANDA_RECVID=$(ssh root@192.168.1.1 -C "ps | grep recv_csi | head -n 1 | cut -d' ' -f2")
    echo "\n"
    echo "Killing process $MIRANDA_RECVID in MirandaTheRouter"

    PHILEAS_RECVID=$(ssh root@192.168.2.1 -C "ps | grep send_Data_con | head -n 1 | cut -d' ' -f2")
    
    echo "\n"
    echo "Killing process $PHILEAS_RECVID in PhileasTheRou  ter"


    PYTHON_RECVID=$(ps | grep run_visualization_server.py | head -n 1 | cut -d' ' -f1)

    ssh root@192.168.1.1 -C "kill -9 ${MIRANDA_RECVID}"
    ssh root@192.168.2.1 -C "kill -9 ${PHILEAS_RECVID}"

    echo "\n"
    echo "Killing process $PYTHON_RECVID in the Python server"

    kill -9 ${PYTHON_RECVID}
    exit
}

trap no_ctrlc SIGINT

while true
do
    sleep 1
done
