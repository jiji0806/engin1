# engin1
1. 
starter_v_h_jw.py
4,5,6,7

starter_v_h_bybit_jw.py
14,15,16,17

2. 
starter_v_h_mj.py
24,25,26,27

3. 
starter_v_h_jkkm.py
34,35,36,37

starter_v_h_bybit_mj.py
44,45,46,47

--------------------------------------------------------
find . -name "*Zone.Identifier" -type f -delete

git clone git@github.com:jiji0806/engin1.git

요약
------------------------------------------------------------
ssh-keygen -f "/root/.ssh/known_hosts" -R "43.200.102.133"
ssh -i "/root/12-16-2024-ourcoffeeforyouj.pem" -o StrictHostKeyChecking=no ubuntu@43.200.102.133
docker exec -it worldengin_container /bin/bash
watch -n 1 'ps -ef | grep live'