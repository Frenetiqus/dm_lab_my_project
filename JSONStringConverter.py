import collections
import json

# This function returns an array containing objects acquired from JSON strings
def getObjectFromJSONDescriptionsString(json_str):
    def create_object(self):
        return collections.namedtuple('object_name', keys)(*values)
    objects = []
    for x in json_str:
        if x == 'missing':
            objects.append(x)
            continue
        a_dict_json = json.loads(x)
        keys = a_dict_json[0].keys()
        values = a_dict_json[0].values()
        an_object = json.loads(x, object_hook=create_object)
        objects.append(an_object[0])
    return objects

# This function returns an array containing objects acquired from JSON strings
def getObjectFromJSONFeaturesString(json_str):
    def create_object(self):
        return collections.namedtuple('object_name', keys)(*values)
    objects = []
    for i in range(len(json_str)):
        if json_str[i] == 'missing':
            objects.append(json_str[i])
            continue
        a_dict_json = json.loads(json_str[i])
        an_item = []
        for x in a_dict_json:
            keys = x.keys()
            values = x.values()
            an_object = json.loads(json_str[i], object_hook=create_object)
            an_item.append(an_object[0])
        objects.append(an_item)
    return objects
