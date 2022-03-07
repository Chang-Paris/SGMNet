import requests
import os
print(os.getcwd())
print(os.path.exists('demo/demo_1.jpg'))

res = requests.post("http://localhost:8080/predictions/superglue/1.0",
                    files=[
                        ('file1', ('demo_1.jpg', open('demo/demo_1.jpg', 'rb'))),
                        ('file2', ('demo_2.jpg', open('demo/demo_2.jpg', 'rb'))),
                        ('kpt1', ('demo1_kpt.npy', open('demo/demo1_kpt.npy', 'rb'))),
                        ('desc1', ('demo1_desc.npy', open('demo/demo1_desc.npy', 'rb'))),
                        ('kpt2', ('demo2_kpt.npy', open('demo/demo2_kpt.npy', 'rb'))),
                        ('desc2', ('demo2_desc.npy', open('demo/demo2_desc.npy', 'rb')))
                           ]
                    )
print(res.text)
#r = requests.post('http://localhost:8080/predictions/superglue/1.0')
#print(r)