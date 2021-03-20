import os

root = os.path.split(os.path.realpath(__file__))[0]
config_json = root + '/tempConfig/config2.json'
coco = root + '/core/data/data/one.names'
path = root + '/core/data/weights6'
anchors = root + '/core/data/data/one_anchors.txt'

"""
dict_keys(['north0024001', 'north0021002', 'north0012002', 'north0015002', 'north0007002', 'north0003002', 'north0027002', 'north0015001', 'north0022002', 'north0002002', 'north0012001', 'north0011001', 'north0005002', 'north0006001', 'north0023002', 'north0014001', 'north0011002', 'north0021001', 'north0004002', 'north0019001', 'north0018001', 'north0018002', 'north0029002', 'north0001001', 'north0016001', 'north0019002', 'north0022001', 'north0030002', 'north0025002', 'north0013002', 'north0028002', 'north0008001', 'north0005001', 'north0008002', 'north0016002', 'north0009001', 'north0024002', 'north0007001', 'north0010002', 'north0028001', 'north0025001', 'north0006002', 'north0029001', 'north0017001', 'north0020002', 'north0013001', 'north0017002', 'north0009002', 'north0010001', 'north0002001', 'north0020001', 'north0003001', 'north0004001', 'north0001002', 'north0027001', 'north0023001', 'north0026001', 'north0030001', 'north0014002', 'north0026002'])
dict_keys(['north0024001', 'north0021002', 'north0012002', 'north0015002', 'north0007002', 'north0003002', 'north0027002', 'north0015001', 'north0022002', 'north0002002', 'north0012001', 'north0011001', 'north0005002', 'north0006001', 'north0023002', 'north0014001', 'north0011002', 'north0021001', 'north0004002', 'north0019001', 'north0018001', 'north0018002', 'north0029002', 'north0001001', 'north0016001', 'north0019002', 'north0022001', 'north0030002', 'north0025002', 'north0013002', 'north0028002', 'north0008001', 'north0005001', 'north0008002', 'north0016002', 'north0009001', 'north0024002', 'north0007001', 'north0010002', 'north0028001', 'north0025001', 'north0006002', 'north0029001', 'north0017001', 'north0020002', 'north0013001', 'north0017002', 'north0009002', 'north0010001', 'north0002001', 'north0020001', 'north0003001', 'north0004001', 'north0001002', 'north0027001', 'north0023001', 'north0026001', 'north0030001', 'north0014002', 'north0026002'])
"""

if __name__ == "__main__":
    print('the root is:', root)
    print('the config_json is:', config_json)
    print('the coco is:', coco)
    print('the path is:', path)
    print('the anchors is:', anchors)
