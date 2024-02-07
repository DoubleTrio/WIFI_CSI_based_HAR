# Before Anything Else:
	- Install Python
	- Install pyqt6
	- Install pyqtgraph
	- Install opencv-python
	- Clone https://github.com/DoubleTrio/WIFI_CSI_based_HAR (We just need the "data_retrieval" folder)
		- Alternate cloning method using sparse checkout for data_retrieval subdirectory:
			- Create empty git repo
				- git init
			- Add repo origin as repo .git url
				- git remote add -f origin  https://github.com/DoubleTrio/WIFI_CSI_based_HAR.git
			- set checkout directory
				- git sparse-checkout set data_retrieval
		- sparse-checkout documentation
			- https://git-scm.com/docs/git-sparse-checkout
	- uncomment line 8: "import cv2" in data_retrieval/run_visualization_server.py

# Running the Entire Setup

1. Plug MirandaTheRouter and PhileasTheRouter in with a charger
2. Connect the computer to MirandaTheRouter with an Ethernet cord
3. Also connect the computer to PhileasTheRouter through WiFi, so change from Ursinus Secure -> PhileasTheRouter (use the WiFi password if necessary)
4. First in the computer terminal, login into MirandaTheRouter with the command `ssh root@192.168.1.1`
5. In another terminal, login into PhileasTheRouter with the command `ssh root@192.168.2.1`
6. In MirandaTheRouter's SSH terminal, run the command `./receive_data.sh <IP_ADDRESS>`. 
    - IP_ADDRESS should be IP address of the computer connected to MirandaTheRouter through the Ethernet cord. In our case, our computer address should most likely look like 192.168.1.`OTHER_NUMBERS`. Example `<IP_ADDRESS>` may look something like `192.168.1.140`  
    - Do note that the shell script defaults to `60000` as the port, you may have to edit the port using the command `vi receive_data.sh` and change the port there
7. In PhileasTheRouter's SSH terminal, run the command `./send_csi_data.sh <PACKET_RATE>`. `<PACKET_RATE>` is the how many packets to send per microsecond. In this case, if `<PACKET_RATE>` is 1000000, then it will send 1 packet per second
8. Open another terminal/console on your computer and run [run_visualization_server.py](https://github.com/DoubleTrio/WIFI_CSI_based_HAR/blob/master/data_retrieval/run_visualization_server.py) with the command `python3 run_visualization_server.py`
    - Do note that the visualization server struggles when the packets are sent at a rate faster than 1 packet per second. Thus, ideally the server code is fixed or rewritten
9. If everything is working, you should be able to see the phase and amplitude visualized real-time!
10. Alternatively, you can run `./run_setup.sh <IP_ADDRESS> <PACKET_RATE>` (you may have to modify the shell script to fit your needs)
