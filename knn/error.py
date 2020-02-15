class Error(Exception):
    pass

class InitializationError(Error):

    def __init__(self, msg):
        self.msg = msg

    def ___str__(self):
        return repr(self.msg)
