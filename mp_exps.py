# from multiprocessing import Pool, cpu_count
# from praatio import tgio
# import time
#
#
# def bla():
#     # tg_gaze = tgio.openTextgrid("gaze_output (1).TextGrid")
#     tg_gaze = tgio.openTextgrid("trott.TextGrid")
#
#     # tg_body = tgio.openTextgrid("body_output (1).TextGrid")
#     # tg_expr = tgio.openTextgrid("expr_output (1).TextGrid")
#     # tg_emotion = tgio.openTextgrid("emotion_output (1).TextGrid")
#     for tier_name in tg_gaze.tierNameList:
#         tier = tg_gaze.tierDict[tier_name]
#         entryList = tier.entryList
#         for idx, entry in enumerate(entryList):
#             print("label", entry.label*100)
#             print("start", entry.start**10)
#             print("end", entry.end**10)
#             print("interval and working on which tier", tier_name, idx+1)
#
#
# def main():
#     start_time = time.time()
#     num_workers = 4
#     assert num_workers > 1, "Need more than 1 worker to reproduce the bug (I think)"
#     pool = Pool(processes=num_workers)
#     pool.map(bla, range(1))
#     print(time.time() - start_time)
#     # bla()
#     # print(time.time() - start_time)
#
#
#
# if __name__ == "__main__":
#     main()
#


from multiprocessing import Process
import sys

import multiprocessing
from multiprocessing import Process


# def print_func(continent='Asia'):
#     return 'The name of continent is : ', continent
#     # print('The name of continent is : ', continent)
#
#
# def print_add_func(continent, add="bla"):
#     return f"{continent} {add}"
    # print(f"{continent} {add}")


# if __name__ == "__main__":  # confirms that the code is under main function
    # names = ['America', 'Europe', 'Africa']
    # name = "Africa"
    # procs = []
    # proc1 = Process(target=print_func)  # instantiating without any argument
    # procs.append(proc1)
    # proc1.start()

    # instantiating process with arguments
    # for name in names:
        # print(name)
    # proc1 = Process(target=print_func, args=(name,))
    # print(proc1)
    # procs.append(proc1)
    # proc1.start()
    # proc2 = Process(target=print_add_func, args=(name, "bruh"))
    # print(proc2)
    # procs.append(proc2)
    # proc2.start()

    # bla = []
    # # complete the processes
    # for proc in procs:
    #     bla.append(proc.join())
    # print(bla)
    # proc1.join()
    # print(bla)
    # print(proc1)
    # proc2.join()
    # print(proc2)

    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # jobs = []
    # jobs.append(proc1)
    # proc1.start()
    # jobs.append(proc2)
    # proc2.start()
    #
    # for proc in jobs:
    #     proc.join()
    # print(return_dict.values())
if __name__ == '__main__':

    import multiprocessing
    import multiprocess as mp
    import defs

    def worker1(v):
        # with v.get_lock():
        #     v.value += 1
        v.value += 1

    # def worker2(v):
    #     with v.get_lock():
    #         v.value += 2

    ctypes_int = mp.Value("i", 0)
    print(ctypes_int.value)

    process1 = mp.Process(
        target=worker1, args=[ctypes_int])
    # process2 = multiprocessing.Process(
    #     target=worker2, args=[ctypes_int])

    process1.start()
    # process2.start()
    process1.join()
    # process2.join()

    print(ctypes_int.value)