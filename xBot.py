import cv2, imutils
# from PIL import Image, ImageStat
import numpy as np
import win32gui, mss, time, os, signal
import ctypes, math
from scipy.spatial import distance as dist
import webcolors

from queue import LifoQueue, Queue
from threading import Thread

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
    return titles

def show_result(img, do):
    while do == True:
        cv2.imshow('OpenCV/Numpy normal', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return

def convert_rgb_to_bgr(img):
    return img[:, :, ::-1]

def grub(window, qIn, x, do):
    print(f'\nStart grubber {x}')
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


# class Gyroscope()

def counter_color(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # removing noise; blur
    # gray = cv2.bilateralFilter(gray, 5, 17, 17)
    # gray = cv2.medianBlur(gray, 5) # fast
    edges = cv2.Canny(gray, 50, 200)

    _, cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in cnts:
        # print("!#len",len(cnt))
        # if len(cnt)<40 or len(cnt)>60: continue
    
        rect = cv2.minAreaRect(cnt)
        print('!#src angle', rect[2])
        box = np.int0(cv2.boxPoints(rect))  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        print('!#area', area)
    
        if area < 150 or area>400: continue
        # if area<15000: continue
    
        # center = (int(rect[0][0]), int(rect[0][1]))
        # вычисление координат двух векторов, являющихся сторонам прямоугольника
        edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
        edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

        if cv2.norm(edge2) < cv2.norm(edge1):
            usedEdge = edge2
        else:
            usedEdge = edge1
    
        reference = (1, 0)  # горизонтальный вектор, задающий горизонт
        angle = 180 / math.pi * math.acos((reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) /
                                          (cv2.norm(reference) * cv2.norm(usedEdge)))
        if angle>90:
            angle = 180 - angle
        else:
            angle = -1 * angle
        img1 = img.copy()
        cv2.drawContours(img1, [cnt], -1, (255, 255, 0), 1)
    
        size = cv2.contourArea(cnt)
        print(size)
    
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
    
        rows, cols = img.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        try:
            cv2.line(img1, (cols - 1, righty), (0, lefty), (0, 255, 0), 1)
        except:
            continue
    
        cv2.drawContours(img1, [box], 0, (0, 255, 0), 2)
        print(rect[1][0])
        im = imutils.rotate(img1, angle)
    
    
        # approx Contours
        # peri = cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        # x, y, w, h = cv2.boundingRect(approx)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
    
    
        cv2.imshow("Game Boy Screen", im)
        # cv2.imshow("Game Boy Screen2", img)
        cv2.waitKey(0)
    return img


class ColorLabeler:
    __slots__ = 'colors', 'lab', 'colorNames'
    
    def __init__(self):
        self.colors = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)}
        self.lab = np.zeros((len(self.colors), 1, 3), dtype="uint8")
        self.colorNames = []
        
        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(self.colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
        
        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
    
    def label(self, image):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        # cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        
        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
        
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)
            
            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                minDist = (d, i)
        
        # return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]
    
    def closest_name(self, rgb):
        min_colours = {}
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - rgb[2]) ** 2
            gd = (g_c - rgb[1]) ** 2
            bd = (b_c - rgb[0]) ** 2
            min_colours[(rd + gd + bd)] = name
        
        return min_colours[min(min_colours.keys())]
    
    def get_html_name(self, rgbIn):
        rgb = (rgbIn[2], rgbIn[1], rgbIn[0])
        try:
            name = webcolors.rgb_to_name(rgb)
        except ValueError:
            name = self.closest_name(rgb)
        return name

class Sensor:
    def __init__(self,roi):
        self.img = roi[0]
        self.startPx = roi[1]
        self.endPx = roi[2]
        self.centerPx = self.get_center()
        # self.lastState = None
        self.reaction = 1
        self.avgColor = self.avg_color()
        self.avgColorTrace = self.avgColor
        self.colorName = self.avg_color_name(self.avgColor)
        self.buffer = Queue()
        self.buffer.maxsize = 6
        while self.buffer.full() is False:
            self.buffer.put(self.avgColor)

    def get_center(self):
        return (round(((self.endPx[0] - self.startPx[0]) / 2) + self.startPx[0]),
                round(((self.endPx[1] - self.startPx[1]) / 2) + self.startPx[1]))

    def avg_color(self):
        """
        return average color for this frame
        :return:
        """
        avgRow = np.average(self.img, axis=0)
        avg = np.average(avgRow, axis=0)
        return [int(avg[0]), int(avg[1]), int(avg[2])]
    
    def avg_color_trace(self):
        """
        Duplicate frames by they buffer position and calc avgColor.
        count = x / position * x for parabola dependency where x - reaction.
	    For parabola dependency newer frame will have maximum influence.
        :return:
        """
        # frames = [[100, 100, 10, 10],
        #           [10, 100, 100, 10],
        #           [10, 10, 100, 100],
        #           [10, 10, 10, 100],
        #           [10, 10, 10, 10],
        #           [10, 10, 10, 10],
        #           ]
        
        counts = []
        for n in range(1, self.buffer.maxsize+1):
            # counts.append(1) # fixed
            # counts.append(round(self.reaction/((n+1)))) # parabola
            counts.append(round((n + 1) / self.reaction))  # linier reverse
        counts = counts[::-1]
        
        frames=list(self.buffer.queue)
        ls = []
        for n, i in enumerate(counts):
            # print(n,i)
            if i==0: break
            while counts[n]!=0:
                ls.append(frames[n])
                counts[n]-=1
        # input()
        if len(ls)==0:
            raise Exception("No frames to calc avgColor. Set right reaction and buffer.")
        avg = np.average(ls, axis=0)
        # avg = (int(avg[0]), int(avg[1]), int(avg[2]), int(avg[3]))
        # input(avg)
        return (int(avg[0]), int(avg[1]), int(avg[2]))
    
    def avg_color_name(self, avgColor):
        return colorLabeler.closest_name(avgColor)
    
    def update(self, img):
        self.img = img[0]
        self.buffer.get()
        self.buffer.task_done()
        self.avgColor = self.avg_color()
        self.buffer.put(self.avgColor)
        self.avgColorTrace = self.avg_color_trace()
        self.colorName = self.avg_color_name(self.avgColorTrace)

# sensors.append([self.cells[i][0], self.cells[i][1], self.cells[i][2],
#                 self.get_cell_center(self.cells[i][1],self.cells[i][2]),
#                 avg, colorLabeler.get_html_name(avg)
#                 ])

class Vision:
    __slots__ = ('imgHeight', 'imgWidth', 'sensors', 'yStep',
            'xStep', 'roiCoordinats', 'roiList', 'roiSensors')
    
    def  __init__(self, img, roi_pos):
        self.imgHeight = img.shape[0]
        self.imgWidth = img.shape[1]
        self.yStep = self.imgHeight // 10
        self.xStep = self.imgWidth // 14
        self.imgHeight -= 10
        self.imgWidth -= 14

        # координаты roi
        self.roiCoordinats = []
        for y in range(0, self.imgHeight, self.yStep):
            for x in range(0, self.imgWidth, self.xStep):
                y1 = y + self.yStep
                x1 = x + self.xStep
                # cell = img[y:y + self.yStep, x:x + self.xStep]
                self.roiCoordinats.append([[x,y], [x1,y1]])

        # весь кадр по частям
        self.roiList = []
        # индекс roi для сенсора
        self.cut_img(img)
        self.roiSensors = [(round(len(self.roiList) * n)) for n in roi_pos]
        
        self.sensors = {}
        for i in self.roiSensors:
            sensor = Sensor(self.roiList[i])
            self.sensors[i] = sensor
     
    # возращает весь кадр по частям
    def cut_img(self, img, all = True):
        if all is False:
            ls = self.roiSensors
        else:
            ls = self.roiCoordinats
        self.roiList.clear()
        for pos in ls:
            # imgGrid =cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0))
            # cell = img[y:y + self.yStep, x:x + self.xStep]
            roi = img[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]
            # show_result(roi, do)
            # time.sleep(0.5)
            # [[imgZone], (stPx), (endPx), (centerPos)]
            self.roiList.append([roi, (pos[0][0], pos[0][1]), (pos[1][0], pos[1][1])])
        # cv2.imwrite(f"asas.png" , img)
    
    # получение конкретных зон, их предобработка
    def update_sensors(self):
        # print(self.sensors[15].img[0]==self.sensors[26].img[0])
        for i in self.roiSensors:
            # show_result(self.roiList[i][0], do)
            # time.sleep(0.5)
            self.sensors[i].update(self.roiList[i])
        # aa=''
    
    def get_sensors(self):
        return self.sensors
        
    def look(self, img):
        self.cut_img(img, all = True)
        self.update_sensors()
        return self.sensors
    
class Stabilizer:
    def stabilize(self,image, old_frame):


            # params for ShiTomasi corner detection
            feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )

            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Create some random colors
            # color = np.random.randint(0,255,(100,3))

            # Take first frame and find corners in it
            # try:
            #     if old_frame!=0:
            #         pass
            #     else:
            #         old_frame = image
            # except Exception as e:
            #     print(e)

            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


            frame = image
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)


            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]


            # Make 3x3 matrix
            h=cv2.findHomography(good_old,good_new)
            #h=cv2.getPerspectiveTransform(good_old,good_new) #not working



            # Now update the previous frame and previous points
            #old_gray = frame_gray.copy()
            #p0 = good_new.reshape(-1,1,2)

            #cv2.destroyAllWindows()

            result=cv2.warpPerspective(frame,h[0], (frame.shape[1],frame.shape[0]))

            return frame, result


signal.signal(signal.SIGTERM, shutdown_me)
signal.signal(signal.SIGINT, shutdown_me)

hwnd = win32gui.FindWindow(None, r'BARRIER X - you are a monster!'
                                 r' (Last level, 1080 60fps).mp4 '
                                 r'- MPC-BE x64 - v1.5.2 (build 3445) beta')
# print(get_windows_titles())
# hwnd = win32gui.FindWindow(None, 'https://player.twitch.tv - Twitch - Mozilla Firefox')
try:
    win32gui.SetForegroundWindow(hwnd)
except Exception:
    print("Не могу захватить экран")
    time.sleep(3)
    os._exit(1)
window = win32gui.GetWindowRect(hwnd)
print('window', window)
colorLabeler = ColorLabeler()
stabilizer = Stabilizer()
font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    qIn = LifoQueue(maxsize=1)
    do=True
    
    Thread(target=grub, args=(window,qIn,0, do,)).start()
    last_img = img = get_img()
    
    # нужно указать в % примерные зоны
    angelsPos = [0.11, 0.185, 0.81, 0.885]
    sensorsPos = [0.415, 0.48,
               0.52, 0.545, 0.55, 0.57,
               0.615, 0.63, 0.665, 0.68,
               0.72, 0.77,
               0.835 , 0.86]
    clear = '\n' * (len(angelsPos) + len(sensorsPos))
    vision = Vision(img, angelsPos + sensorsPos)
    
    # после инциаоизации можно получить реальные позиции
    angelsPos, sensorsPos = vision.roiSensors[:4], vision.roiSensors[4:]
    
    while True:
        # time.sleep(0.1)
        st = time.time()
        img = qIn.get()
        imgCanny = counter_color(img)
        # input()
        # try:
        #     img, result = stabilizer.stabilize(img, last_img)
        # except:
        #     result = img
        sensors = vision.look(img)
        # res, mas = rgb_to_hsv(img)
    
        inGame = False
        for n, i in enumerate(angelsPos):
            if sensors[i].colorName not in ['black', 'maroon']:
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
    
        show_result(imgCanny, do)
        qIn.task_done()
        statistic = f'{clear}  fps: {round(1 / (time.time()-st),1)} In game: {inGame}' \
              f'\n  angle[0]: {sensors[angelsPos[0]].colorName}' \
              f'\n  angle[1]: {sensors[angelsPos[1]].colorName}' \
              f'\n  angle[2]: {sensors[angelsPos[2]].colorName}' \
              f'\n  angle[3]: {sensors[angelsPos[3]].colorName}'

        for n, i in enumerate(sensorsPos):
            statistic += f'\n  sens[{n}]: {sensors[i].colorName}'
        print(statistic)
        last_img = img
    do = False
