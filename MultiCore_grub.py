class mlt():
    from multiprocessing import Process, Queue
    q = Queue(maxsize=2)
    with mss.mss() as sct:
        window = sct.monitors[1]
        processes = [Process(target=grub, args=(window, q, x,)) for x in range(5)]
        for p in processes: p.start()
        n=0
        st0 = time.time()
        while n<100:
            st = time.time()
            cv2.imshow('OpenCV/Numpy normal', q.get())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            print(f'\rfps: {1 / (time.time()-st)}', end='')
            st = time.time()
            n+=1
        print('\nstop show', time.time()-st0)
        for p in processes: p.join()
    cv2.destroyAllWindows()
