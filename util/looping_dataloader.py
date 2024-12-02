from torch.utils.data import DataLoader


class LoopingDataLoader[T](DataLoader[T]):
    def __init__(self, loader: DataLoader[T], max_iter: int | None = None):
        self.loader = loader
        self.max_iter = max_iter
        self.iterator = iter(self.loader)

    def get_next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def __iter__(self):
        count = 0
        while self.max_iter is None or count < self.max_iter:
            yield self.get_next()
            count += 1

    def __len__(self):
        return self.max_iter if self.max_iter is not None else float("inf")
