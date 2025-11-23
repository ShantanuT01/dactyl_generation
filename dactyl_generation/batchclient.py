from abc import ABC
from abc import abstractmethod

class BatchClient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_batch_job(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_batch_job_output(self, *args, **kwargs):
        pass