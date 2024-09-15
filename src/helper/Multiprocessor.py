import cv2
import multiprocessing as mp

class Multiprocessor:

    def __init__(self, NUM_PROCESS: int):
        self.NUM_PROCESS = NUM_PROCESS
    
    def run(self, data, function) -> list: # в data может быть ndarray или list, в function - функция
        new_data = [None] * len(data)
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        processes = [
            mp.Process(target=self._run_deamon, 
                       args=(input_queue, output_queue, function, i + 1), daemon=True)
            for i in range(self.NUM_PROCESS)
            ]
        
        for process in processes:
            process.start()
            
        for key, element in enumerate(data):
            input_queue.put((key, element))
            
        k = 0   
        key = None
        element = None
        while k < len(data):
            if not output_queue.empty():
                key, element = output_queue.get()
                new_data[key] = element
                k += 1
            
        for process in processes:
            process.terminate()
            
        return new_data
    
    def _run_deamon(self, input_queue: mp.Queue, output_queue: mp.Queue,
                         function, index_process: int) -> None:
        while True:
            if not input_queue.empty():
                key, input_data = input_queue.get()
                output_data = function(input_data, index_process)
                output_queue.put((key, output_data))