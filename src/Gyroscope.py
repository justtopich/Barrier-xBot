########################
# Сamera angle determination.
# Used to align horizon before sensors processing.
# Like a sensors gyroscope have inertia and reaction of results changes.
# It allow to cover some inaccuracies in processing.
########################

import time

class Gyroscope():
    __slots__ = ['img', 'imgHeight', 'imgWidth', 'imgArea', 'timerAbsCenter',
                 'minArea', 'maxArea', 'roiPos', 'buffer', 'timerRoiCenter',
                 'reference', 'angels', 'horizon', 'digits', 'roi',
                 'imgCenter', 'roiWeight', 'roiHeight', 'roiSelfCenter',
                 'imgShades']

    def __init__(self, img, settings):
        print('Gyroscope initialization')
        self.img = img
        self.imgShades = None   # быстрое выделение белый объектов, хранит для Vision
        self.imgHeight = img.shape[0]
        self.imgWidth = img.shape[1]
        self.imgCenter = [round(self.imgWidth / 2), round(self.imgHeight / 2)]
        # для фильтра по площади
        self.imgArea = self.imgWidth * self.imgHeight
        self.minArea = self.imgArea * settings['gyroscope']['digitMinArea']
        self.maxArea = self.imgArea * settings['gyroscope']['digitMaxArea']
        # константы для смещения зоны захвата
        self.roiSelfCenter = None  # центр относительно самой зоны
        self.roiHeight = round(self.imgHeight * 0.3)
        self.roiWeight = round(self.imgWidth * 0.5)
        self.roiPos = self.get_roi_pos()
        self.roi = self.get_roi()
        self.reference = (1, 0)  # горизонтальный вектор, задающий горизонт
        self.angels = [0, 0, 0]
        self.horizon = 0  # avg по всем цифрам
        self.buffer = Queue()
        self.buffer.maxsize = settings['gyroscope']['bufferSize']
        self.digits = {}
        self.timerRoiCenter = None  # центр таймера внутри зоны
        self.timerAbsCenter = self.imgCenter  # центр таймера в исходном кадре
        while self.buffer.full() is False:
            self.buffer.put((self.horizon,self.timerAbsCenter))

    # получение координат стартовой зоны
    def get_roi_pos(self):
        self.horizon = 0
        # x = round(self.imgWidth * -0.28 / 2 + self.imgCenter[0])
        # y = round(self.imgHeight * -0.4 / 2 + self.imgCenter[1])
        x = round(self.imgWidth * -0.5 / 2 + self.imgCenter[0])
        y = round(self.imgHeight * -0.4 / 2 + self.imgCenter[1])
        x1 = x + self.roiWeight
        y1 = y + self.roiHeight
        self.roiSelfCenter = self.get_center(x, y, x1 - x, y1 - y)

        # # отладка
        # cv2.rectangle(self.img, (x, y), (x1, y1), (0, 255, 0))
        # cv2.imshow("Gyroscope", self.img)
        # cv2.waitKey(1)
        # print('')

        return ([[x, y], [x1, y1]])

    # обновление координат зоны захвата
    def update_roi_pos(self):
        '''
        считается от центра таймера, таким образом
        зона всегда будет следовать за таймером
        :return:
        '''
        x = int(self.roiSelfCenter[0] - self.imgWidth * 0.15 / 2)
        y = int(self.roiSelfCenter[1] - self.imgHeight * 0.35 / 2)
        x1 = x + self.roiWeight
        y1 = y + self.roiHeight
        self.roiPos = ([[x, y], [x1, y1]])

        # отладка
        # cv2.rectangle(self.img, (x, y), (x1, y1), (0, 255, 0))
        # cv2.imshow("Gyroscop-Track", self.img)
        # cv2.waitKey(1)
        # print('')

    # получение самой зоны
    def get_roi(self):
        return self.img[self.roiPos[0][1]:self.roiPos[1][1],
               self.roiPos[0][0]:self.roiPos[1][0]]

    def find_digits(self):
        # gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        gray = color_rgb_filter(self.roi, 160)

        # gray = cv2.inRange(self.roi,np.array([180,180,180]),np.array([255,255,255]))
        # self.imgShades = color_rgb_filter(self.img)

        # cv2.imshow("Gyroscope-in", gray)
        # cv2.waitKey(1)

        edges = cv2.Canny(gray, 100, 200)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # dilated = cv2.dilate(edges, kernel)

        # use _,cnts,_ for old versions
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self.angels.clear()
        self.digits.clear()

        idx = -1
        distIdx = {}
        distList = []
        for cnt in cnts:
            aa = len(cnt)
            # x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(self.roi, (x, y), (x + w, y + h), (155, 150, 0), 1)
            # cv2.imshow("Gyroscope-angels-end", self.roi)
            # cv2.waitKey(1)
            # print('')

            # группировка близких точек
            approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
            # фильтр по выходу вершины за зону
            try:
                for px in approx:
                    assert 2 < px[0][0] < self.roiWeight - 2
                    assert 2 < px[0][1] < self.roiHeight - 2
            except:
                continue

            # фильтр по количеству вершин
            a = len(approx)
            if len(approx) < 6: continue

            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rect))  # округление координат

            # фильтр по площади
            area = int(rect[1][0] * rect[1][1])  # вычисление площади
            if area < self.minArea or area > self.maxArea: continue

            # координаты двух векторов, являющихся сторонам прямоугольника
            # edge1 = (box[1][0] - box[0][0], box[1][1] - box[0][1])
            # edge2 = (box[2][0] - box[1][0], box[2][1] - box[1][1])
            edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
            edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
            # длина сторон
            edgeNorm1 = cv2.norm(edge1)
            edgeNorm2 = cv2.norm(edge2)
            if edgeNorm2 < 1: continue

            # фильтр по пропорциям
            ratio = edgeNorm1 / edgeNorm2
            if not ((0.28 < ratio < 0.9) or (1.44 < ratio < 2.7)): continue

            # for i in approx:
            #     cv2.circle(self.roi, tuple(i[0]), 3, (0, 255, 0), 1)
                # cv2.imshow("Gyroscope=apr", self.roi)
            #     cv2.waitKey(1)
            #     print('')

            # поиск большей стороны
            if edgeNorm2 < edgeNorm1:
                usedEdge = edge2
            else:
                usedEdge = edge1

            angle = 180 / math.pi * math.acos((self.reference[0] * usedEdge[0] +
                                               self.reference[1] * usedEdge[1]) /
                                              (cv2.norm(self.reference) * cv2.norm(usedEdge)))
            # фильтр по наклону
            if (-50 < angle < -140) or (50 < angle < 140): continue

            # print('myAngle0', angle)
            # определение направления поворота
            if angle > 90:
                angle = 180 - angle
            else:
                angle = -1 * angle

            # print('myAngle', angle)
            self.angels.append(angle)

            # подсчёт центра данной цифры
            idx += 1
            x, y, w, h = cv2.boundingRect(cnt)
            center = self.get_center(x, y, w, h)

            # подсчёт расстояний до центров других цифр
            for d in self.digits:
                if d != idx:
                    s = f'{d}-{idx}'
                    D = dist.euclidean(center, self.digits[d]['center'])
                    distList.append(D)
                    distIdx[s] = D

            self.digits[idx] = {'cnt' : cnt,
                                'center' : center,
                                'area' : area,
                                'ratio' : ratio,
                                'angle' : angle,
                                'approx' : approx,
                                # 'height' : cv2.norm(usedEdge)}
                                'height' : math.fabs(usedEdge[0])}
            # center.append(cv2.boundingRect(cnt))

            # для отладки
            # x, y, w, h = cv2.boundingRect(box)
            # cv2.rectangle(self.roi, (x, y), (x + w, y + h), (255, 255, 0), 1)
            # cv2.imshow("Gyroscope-angels-end", self.roi)
            # cv2.waitKey(1)
            # print('')
            # cv2.circle(self.roi, center, 6, (255, 255, 255), 1)   # центр бокса
            #
            # rows, cols = im.shape[:2]
            # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            # lefty = int((-x * vy / vx) + y)
            # righty = int(((cols - x) * vy / vx) + y)
            # try:
            #     cv2.line(im, (cols - 1, righty), (0, lefty), (255, 255, 0), 1)
            # except:
            #     continue
            #
            # print('!#myAngle', angle)
            # im = imutils.rotate(im, angle)

        # исключает слишком удалённые от таймера обекты
        if idx > 0:
            try:
                # avgDist = np.average(distList, axis=0)
                avgDist = np.average([self.digits[i]['center'] for i in self.digits], axis=1).tolist()
                # mid = len(avgDist) // 2
                # avgDist = avgDist[mid:] + avgDist[:mid]
                avgDist1 = self.reject_outliers(avgDist, m=2.25)
                for n,i in enumerate(avgDist):
                    x, y, w, h = cv2.boundingRect(self.digits[n]['cnt'])
                    if i not in avgDist1:
                        cv2.rectangle(self.roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(self.roi, (x, y), (x + w, y + h), (255, 255, 0), 1)

            except Exception as e:
                print(e)

        # cv2.imshow("Gyroscope", self.roi)
        # cv2.waitKey(1)
        # print(';')

        # для отладки
        # im = self.roi.copy()
        # self.get_horizon()
        # print("!#avgAngle", self.horizon)
        #
        # im = imutils.rotate(im, self.horizon)
        # cv2.imshow("Gyroscope", im)
        # cv2.waitKey(1)

    def reject_outliers(self, data, m=2.):
        # if not isinstance(data, np.ndarray):
        # data1 = [i for i in data]
        # data2 = [round(i * 1.1 ,1) for i in data]
        # data = data2
        data = np.array(data)
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 1.
        res = data[s < m]
        if not isinstance(res[0],np.float64):
            return res[0]
        return res
        # else:
        #     d = np.abs(data - np.median(data))
        #     mdev = np.median(d)
        #     s = d / mdev if mdev else 0.
        #     return data[s < m]

    # поиск среднего угла поворота исключая выбросы
    def get_horizon(self):
        """
        1. сортировка
        2. поиск медианы
        3. удаление выбросов
        4. нахождение средней
        :return:
        """

        # TODO можно считать по 2 цифрам, если их наклон будет совпадать
        # TODO но для этого надо считать разницу их наклона между собой

        # обновление горизонтали только при нахождении не менее 3 цифр
        # if not (2 < len(self.angels) < 7):
        # aa = len(self.digits)
        if  2 > len(self.digits):
            return
        else:
            self.get_timer_center()
            self.horizon = np.average(self.reject_outliers(self.angels), axis=0)

            # абсолютный центр таймера
            x = self.roiPos[0][0] + self.timerRoiCenter[0]
            y = self.roiPos[0][1] + self.timerRoiCenter[1]

            # смещение таймера в кадре
            # try:
            #     x1 = self.timerAbsCenter[0] - x
            #     y1 = self.timerAbsCenter[1] - y
            #     self.roiPos[0][0] -= x1
            #     self.roiPos[0][1] -= y1
            #     self.roiPos[1][0] -= x1
            #     self.roiPos[1][1] -= y1
            # except:
            #     pass

            self.timerAbsCenter = (x,y)

            # cv2.circle(self.img, self.timerAbsCenter, 5, (0, 255, 255), -1)
            # cv2.imshow("Gyroscope1", self.img)
            # cv2.waitKey(1)
            # print('')

    # пока fixed можно не использовать
    def avg_trace(self):
        angle = []
        center = []
        for frame in self.buffer.queue:
            angle.append(frame[0])
            center.append(frame[1])
        self.horizon = np.average(angle, axis=0)
        self.timerAbsCenter = np.average(center, axis=0)
        self.timerAbsCenter = (int(self.timerAbsCenter[0]),int(self.timerAbsCenter[1]))

    def get_center(self, x, y, w, h):
        return (round(w / 2 + x), round(h / 2 + y))

    def get_timer_center(self):
        '''
        Самый верный способ найти точку схождения дорог - поиск
        середины всех цифр, т.к. она всегда находятся под ними.
        '''
        # TODO делать смещения если найдены только крайние цифры
        # self.digits = sorted(self.digits)

        x = np.average([self.digits[i]['center'][0] for i in self.digits], axis=0)
        y = np.average([self.digits[i]['center'][1] for i in self.digits], axis=0)

        # y = np.average([self.digits[0][1],self.digits[-1][1]], axis=0)
        # self.digits = sorted(self.digits, key=lambda x: int(x[0]))
        # x = np.average([self.digits[0][0],self.digits[-1][0]], axis = 0)

        # оставить прежнее значение
        try:
            self.timerRoiCenter = (int(x), int(y))
        except:
            pass

    def update(self, img):
        self.img = img
        self.roi = self.get_roi()
        self.find_digits()
        # после нахождения цифр есть
        # их углы наклона и точки центров
        # self.update_roi_pos()
        self.get_horizon()
        self.buffer.get()
        self.buffer.task_done()
        self.buffer.put((self.horizon,self.timerAbsCenter))
        self.avg_trace()

        # отладка
        # print('!#roiPos',self.roiPos)

        return self.horizon

if __name__ is '__main__':
    raise Exception('Use xBot to start')
else:
    from __main__ import np, cv2, math, Queue, color_rgb_filter, dist