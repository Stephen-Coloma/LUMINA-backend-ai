import torch, time, gc

start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print('\n' + local_msg)
    print('Total execution time = {:.3f} sec'.format(end_time - start_time))
    print('Max memory used by tensors = {} bytes'.format(torch.cuda.max_memory_allocated()))
