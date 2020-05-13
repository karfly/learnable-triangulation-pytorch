# From https://stackoverflow.com/questions/51562221/python-multiprocessing-overflowerrorcannot-serialize-a-bytes-object-larger-t

# TODO: Use torch.multiprocessing or not?
from multiprocessing.reduction import ForkingPickler, AbstractReducer

class ForkingPickler4(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=4):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=4):
    ForkingPickler4(file, protocol).dump(obj)

class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump
