from queue import LifoQueue, Queue
from threading import Thread
import time, os, signal
import ctypes

import win32gui, mss, win32api, win32con

import cv2, imutils
import numpy as np
import math
import webcolors
from scipy.spatial import distance as dist

safety = {6: ['coral', 'tomato', 'lightpink', 'sandybrown', 'lightsalmon', 'darkorange'],
          4: ['indianred', 'lightcoral', 'crimson', 'salmon', 'chocolate', 'orangered' ],
          2: ['brown', 'polevioletred', 'mediumvioletred', 'sienna', 'deeppink'],
          1: ['indigo', 'midnightblue', 'darkslateblue', 'darkslategrey', 'purple', 'dimgrey', 'novy', 'black']}


def shutdown_me(signal, frame):
    os._exit(1)


def get_windows_titles():
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible

    titles = []

    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            titles.append(buff.value)
        return True

    EnumWindows(EnumWindowsProc(foreach_window), 0)
    # print(titles)
    return titles


def get_window(name):
    titles = get_windows_titles()
    for i in titles:
        if name in i:
            return i
    raise Exception("Nothing to capture")


def show_result(img, do):
    while do == True:
        cv2.imshow('OpenCV/Numpy normal', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return


def convert_rgb_to_bgr(img):
    return img[:, :, ::-1]


# поиск белых цветов
def color_rgb_filter(image, min):
    """
    Позволяет выделять цвета не переводя в HSV формат
    за счёт чего не снижается производительность
    :param image:
    :return:
    """
    B, G, R, _ = cv2.split(image)

    # R = np.where(R < 160, 0, R)
    # G = np.where(G < 160, 0, G)
    # B = np.where(B < 160, 0, B)
    #
    # R = np.where(G == 0, 0, R)
    # R = np.where(B == 0, 0, R)
    # G = np.where(R == 0, 0, G)
    # B = np.where(R == 0, 0, B)

    R = np.where(R < min, 0, R)
    # R[R > 179] = 255
    R = np.where(G < min, 0, R)
    R = np.where(B < min, 0, R)

    return cv2.merge([R])
    # return cv2.merge([B,G,R])


def grub(window, qIn, x, do):
    print(f'Start grubber {x}')
    with mss.mss() as sct:
        while do is True:
            # imgSrt = sct.grab(window)
            img = sct.grab(window)
            # img = Image.frombytes('RGB', imgSrt.size, imgSrt.rgb)
            img = np.array(img)

            try:
                qIn.put(img)
            except:
                pass
    print(f'\nStop grubber {x}')

def get_img():
    img = qIn.get()
    qIn.task_done()
    return img

def rgb_to_hsv(img):
    img = convert_rgb_to_bgr(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([120,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res, mask

def xbot():
    # showStat = 0
    while True:
        # showStat += 1
        st = time.time()
        # time.sleep(0.1)
        img = qIn.get()
        # imgCanny = counter_color(img)
        horizon = gyroscope.update(img)
        img = imutils.rotate(img, horizon)
        # input()
        # try:
        #     img, result = stabilizer.stabilize(img, lastImg)
        # except:
        #     result = img
        # print('sens')
        sensors = vision.look(img)
        # res, mas = rgb_to_hsv(img)

        inGame = False
        for n, i in enumerate(angelsPos):
            if sensors[i].colorName not in ['black', 'maroon', 'dimgrey', 'darkolivegreen']:
                inGame = True
            cv2.rectangle(img, sensors[i].startPx, sensors[i].endPx, (255, 255, 255))
            cv2.circle(img, sensors[i].centerPx, 9, sensors[i].avgColorTrace, -1)
            cv2.circle(img, sensors[i].centerPx, 10, (255, 255, 255), 1)
            cv2.putText(img, f'{n}', (sensors[i].startPx[0], sensors[i].endPx[1]),
                        font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for n, i in enumerate(sensorsPos):
            cv2.rectangle(img, sensors[i].startPx, sensors[i].endPx, (255, 255, 255))
            cv2.circle(img, sensors[i].centerPx, 9, sensors[i].avgColorTrace, -1)
            cv2.circle(img, sensors[i].centerPx, 10, (255, 255, 255), 1)
            cv2.putText(img, f'{n}', (sensors[i].startPx[0], sensors[i].endPx[1]),
                        font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # for i in vision.plane['sensors']:
        #     cv2.circle(img, vision.plane['sensors'][i].centerPx, 9, vision.plane['sensors'][i].avgColorTrace, -1)

        if inGame is False:
            gyroscope.get_roi_pos()

        qIn.task_done()
        # if showStat > 2:
        stat = f'{clear}  fps: {round(1 / (time.time()-st),1)} In game: {inGame}' \
            f'\n  horizon: {horizon}' \
            f'\n  angle[0]: {sensors[angelsPos[0]].colorName}' \
            f'\n  angle[1]: {sensors[angelsPos[1]].colorName}' \
            f'\n  angle[2]: {sensors[angelsPos[2]].colorName}' \
            f'\n  angle[3]: {sensors[angelsPos[3]].colorName}'

        for i in (vision.plane['sensors']):
            stat += f'\n  sens[{i}]: {vision.plane["sensors"][i].safety} {vision.plane["sensors"][i].colorName}'

        # print(stat)
        d = 11
        for text in stat[5:].split('\n '):
            cv2.putText(img, text, (10, sensors[angelsPos[0]].endPx[1] + d),
                        font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            d += 12
        # showStat = 0
        controller.play(vision.plane['sensors'])
        show_result(img, do)

if __name__ == "__main__":
    # Windows запускает модули exe из папки пользователя
    # Папка должна определяться только исполняемым файлом
    keys = os.path.split(os.path.abspath(os.path.join(os.curdir, __file__)))
    homeDir = keys[0].replace('\\', '/') + '/'
    appName = keys[1][:keys[1].find('.')].lower()

    from config import settings
    from Gyroscope import Gyroscope
    from Sensor import Sensor
    from Vision import Vision
    from Controller import Controller

    signal.signal(signal.SIGTERM, shutdown_me)
    signal.signal(signal.SIGINT, shutdown_me)

    target = get_window(settings['general']['target'])
    hwnd = win32gui.FindWindow(None, target)

    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        print(f"Не могу захватить экран: {e}")
        time.sleep(3)
        os._exit(1)
    window = win32gui.GetWindowRect(hwnd)
    print('window', window)
    # stabilizer = Stabilizer()

    # enter = 77
    win32gui.SetForegroundWindow(hwnd)
    # win32api.keybd_event(18, 0, 0, 0)
    # win32api.keybd_event(win32con.SHIFT_PRESSED, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
    controller = Controller()

    font = cv2.FONT_HERSHEY_SIMPLEX

    qIn = LifoQueue(maxsize=1)
    do=True

    Thread(target=grub, args=(window,qIn,0, do,)).start()
    lastImg = img = get_img()

    gyroscope = Gyroscope(img,settings)

    # нужно указать в % примерные зоны
    angelsPos = settings['sensors']['angelsPos']
    sensorsPos = settings['sensors']['sensorsPos']
    clear = '\n' * (len(angelsPos) + len(sensorsPos))

    vision = Vision(img,angelsPos + sensorsPos,settings,gyroscope)
    if settings['sensors']['showGrid'] == True:
        vision.cut_img = vision.cut_img_deb

    # после инциаоизации можно получить реальные позиции
    angelsPos, sensorsPos = vision.roiSensors[:4], vision.roiSensors[4:]

    xbot()
    do = False
