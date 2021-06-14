import numpy as np
import time

from mvn.utils.misc import is_master


class MiniMonValue:
    def __init__(self):
        self.runtime_sum = np.float32(0)
        self.runtime_min = np.float32('inf')
        self.runtime_max = -np.float32('inf')
        self.num = 0
        self.runtime_last = -1

    def apply(self, value):
        self.runtime_last = value
        self.runtime_sum += value
        self.runtime_min = min(self.runtime_min, value)
        self.runtime_max = max(self.runtime_max, value)
        self.num += 1

    def get_avg(self):
        return self.runtime_sum / self.num

    def get_min(self):
        return self.runtime_min

    def get_max(self):
        return self.runtime_max

    def get_last(self):
        return self.runtime_last

    def get_stats(self):
        return {
            'min': self.get_min(),
            'max': self.get_max(),
            'avg': self.get_avg(),
            'last': self.get_last(),
            'count': self.num
        }


class MiniMon:
    def __init__(self, only_by_master=True, only_master_should_print=True):
        self.store = {}  # str -> MiniMonValue
        self.entries = []  # LIFO
        self.only_by_master = only_by_master
        self.only_master_should_print = only_master_should_print

    def _can_do_transaction(self):
        if self.only_by_master:
            return is_master()

        return True

    def _can_print(self):
        if self.only_master_should_print:
            return is_master()

        return True

    def get_time_now(self):
        return time.time()

    def enter(self):
        if self._can_do_transaction():
            self.entries.append(self.get_time_now())

    def leave(self, checkpoint_name):
        if self._can_do_transaction():
            last_time = self.entries.pop()

            if checkpoint_name not in self.store:
                self.store[checkpoint_name] = MiniMonValue()

            self.store[checkpoint_name].apply(self.get_time_now() - last_time)

    def get_stats(self):
        return {
            checkpoint: info.get_stats()
            for checkpoint, info in self.store.items()
        }

    def print_stats(self, as_minutes=False):
        if self._can_print():
            f_out = '{:>20} x {:10d} ~ {:10.1f} [min: {:10.1f},    max: {:10.1f},    last: {:10.1f}]'

            for checkpoint, info in sorted(self.store.items(), key=lambda x: x[1].get_avg()):
                if info.num > 1:
                    print(f_out.format(
                        checkpoint[-20:],
                        info.num,
                        info.get_avg() / (60.0 if as_minutes else 1.0),
                        info.get_min() / (60.0 if as_minutes else 1.0),
                        info.get_max() / (60.0 if as_minutes else 1.0),
                        info.get_last() / (60.0 if as_minutes else 1.0),
                    ))
                else:  # single call
                    print('{:>20}   {:>10} ~ {:10.1f}'.format(
                        checkpoint[-20:],
                        '',
                        info.get_avg() / (60.0 if as_minutes else 1.0)
                    ))

            if as_minutes:
                print('    all times are in minutes')
            else:
                print('    all times are in seconds')
