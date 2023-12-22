import heapq


class HeapPriorityQueue:
    def __init__(self):
        self.queue = []
        self.size = 0

    def put(self, value):
        self.push(value)

    def push(self, value): #, priority=None):
        # if priority is None:
        #     priority = 0
        # heapq.heappush(self.queue, (priority, self.seq, value))
        heapq.heappush(self.queue, value)
        self.size += 1

    def get(self):
        return self.pop()

    def pop(self):
        # item = heapq.heappop(self.queue)[-1]
        item = heapq.heappop(self.queue)
        self.size -= 1
        return item

    def qsize(self):
        return self.size

    def empty(self):
        return self.size == 0

    def __len__(self):
        return self.size

    def __bool__(self):
        return True if self.queue else False
