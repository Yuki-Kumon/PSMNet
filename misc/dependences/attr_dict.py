import yaml


class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def __str__(self):
        return 'Attribute Dictionary:\n{}'.format(self.obj)

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()

    def save(self, path):
        """
        path (str) : the path that saves config
        """
        with open(str(path), "w") as f:
            yaml.dump(self.obj, f)


if __name__ == '__main__':
    # f = open('configs/example.yml', "r")
    f = open('configs/configs.yml', 'r')
    config = AttributeDict(yaml.load(f))
    print(config.fields()['dataset']['original']['path'])
