import multiprocessing
import cv2


def cam_loop(queue_from_cam):
    cap = cv2.VideoCapture(0)
    while True:
        hello, img = cap.read()
        queue_from_cam.put(img)


def show_bw(queue_from_cam):
    while True:
        if queue_from_cam.empty():
            continue
        from_queue = queue_from_cam.get()
        print ("wwww")
        cv2.waitKey(10)
        cv2.imshow("bw", cv2.cvtColor(from_queue, cv2.COLOR_BGR2GRAY))


def main():
    print('initializing cam')
    queue_from_cam = multiprocessing.Queue()
    gaze_process = multiprocessing.Process(target=cam_loop, args=(queue_from_cam,))
    gaze_process.start()
    # cam_process = multiprocessing.Process(target=cam_loop, args=(queue_from_cam,))
    # show_bw_proccess = multiprocessing.Process(target=show_bw, args=(queue_from_cam,))
    # cam_process.start()
    # show_bw_proccess.start()
    # while True:
    #     if queue_from_cam.empty():
    #         continue
    #     from_queue = queue_from_cam.get()
    #     cv2.imshow("from queue", from_queue)
    #     key = cv2.waitKey(1)
    #     if key == ord("q"):
    #         cv2.destroyAllWindows()
    #         break
    # print("Destroying process...")
    # cam_process.terminate()
    # cam_process.join()


if __name__ == '__main__':
    main()