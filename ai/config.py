import os

root = os.path.split(os.path.realpath(__file__))[0]
config_json = root + '/tempConfig/config.json'
coco = root + '/core/data/data/one.names'
path = root + '/core/data/weights6'
anchors = root + '/core/data/data/one_anchors.txt'

if __name__ == "__main__":
    print('the root is:', root)
    print('the config_json is:', config_json)
    print('the coco is:', coco)
    print('the path is:', path)
    print('the anchors is:', anchors)
