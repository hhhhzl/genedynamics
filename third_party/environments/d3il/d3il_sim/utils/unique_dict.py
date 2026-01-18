class UniqueDict(dict):
    def __init__(self, err_msg: str = None):
        if err_msg is None:
            err_msg = "Key already exists:"

        self.err_msg = err_msg

    def __setitem__(self, key, value):
        if key not in self:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError("{} {}".format(self.err_msg, key))
