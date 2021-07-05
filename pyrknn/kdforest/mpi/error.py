class Error(Exception):
    "Simple Error Class"
    pass

class InitializationError(Error):
    "Error Type to specify incorrect configuration parameters"
    def __init__(self, msg):
        "Initialize error message"
        self.msg = msg

    def ___str__(self):
        "Print error message"
        return repr(self.msg)
