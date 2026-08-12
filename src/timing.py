import time as time


class Timer:
    def __init__(self, desc: str='Process done', ndigits=4, active: bool=True):
        self.desc = desc
        self.ndigits = ndigits
        self.active = active
        

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            total_time = round(time.perf_counter() - self.start, self.ndigits)
            print(f'{self.desc} in {total_time} seconds.')

            