########################
# Сamera angle determination.
# Used to align horizon before sensors processing.
# Like a sensors gyroscope have inertia and reaction of results changes.
# It allow to cover some inaccuracies in processing.
########################


class Gyroscope():
    __slots__ = ['img', 'imgHeight', 'imgWidth', 'imgArea', 'timerAbsCenter',
                 'minArea', 'maxArea', 'roiPos', 'buffer', 'timerRoiCenter',
                 'reference', 'angels', 'horizon', 'digitsCenter', 'roi',
                 'imgCenter', 'roiWeight', 'roiHeight', 'roiSelfCenter',
                 'imgShades']

    def __init__(self, img, settings):
        self.img = img
        self.imgShades = None   # быстрое выделение белый объектов, хранит для Vision
        self.imgHeight = img.shape[0]
        self.imgWidth = img.shape[1]
        self.imgCenter = [round(self.imgWidth / 2), round(self.imgHeight / 2)]
        # для фильтра по площади
        self.imgArea = self.imgWidth * self.imgHeight
        self.minArea = self.imgArea * 0.00016
        self.maxArea = self.imgArea * 0.0013
        # константы для смещения зоны захвата
        self.roiSelfCenter = None  # центр относительно самой зоны
        self.roiHeight = round(self.imgHeight * 0.18)
        self.roiWeight = round(self.imgWidth * 0.20)
        self.roiPos = self.get_roi_pos()
        self.roi = self.get_roi()
        self.reference = (1, 0)  # горизонтальный вектор, задающий горизонт
        self.angels = [0, 0, 0]
        self.horizon = 0  # avg по всем цифрам
        self.buffer = Queue()
        self.buffer.maxsize = settings['gyroscope']['bufferSize']
        self.digitsCenter = []
        self.timerRoiCenter = None  # центр таймера внутри зоны
        self.timerAbsCenter = None  # центр таймера в исходном кадре
        while self.buffer.full() is False:
            self.buffer.put(self.horizon)

    # получение координат стартовой зоны
    def get_roi_pos(self):
        self.horizon = 0
        # x = round(self.imgWidth * -0.28 / 2 + self.imgCenter[0])
        # y = round(self.imgHeight * -0.4 / 2 + self.imgCenter[1])
        x = round(self.imgWidth * -0.15 / 2 + self.imgCenter[0])
        y = round(self.imgHeight * -0.35 / 2 + self.imgCenter[1])
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
        cv2.rectangle(self.img, (x, y), (x1, y1), (0, 255, 0))
        cv2.imshow("Gyroscop-Track", self.img)
        cv2.waitKey(1)
        print('')

    # получение самой зоны
    def get_roi(self):
        return self.img[self.roiPos[0][1]:self.roiPos[1][1],
               self.roiPos[0][0]:self.roiPos[1][0]]

    def get_angels(self):
        # gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        gray = color_rgb_filter(self.roi)
        # self.imgShades = color_rgb_filter(self.img)

        # cv2.imshow("Gyroscope-in", gray)
        # cv2.imshow("Gyroscope2", self.imgShades)
        # cv2.waitKey(1)

        edges = cv2.Canny(gray, 100, 200)
        # use _,cnts,_ for old versions
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.angels.clear()
        self.digitsCenter.clear()
        idx = 0
        for cnt in cnts:
            # фильтр по выходу вершины за зону
            approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
            try:
                for px in approx:
                    assert 2 < px[0][0] < self.roiWeight - 2
                    assert 2 < px[0][1] < self.roiHeight - 2
            except:
                continue

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
            # print('!#ratio', ratio)
            if not ((0.28 < ratio < 0.9) or (1.44 < ratio < 2.7)):
                continue

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
            x, y, w, h = cv2.boundingRect(cnt)
            center = list(self.get_center(x, y, w, h))
            center.append(idx)
            center.append(cv2.boundingRect(cnt))
            idx += 1
            self.digitsCenter.append(center)
            # return

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
        try:
            ls = []
            for m in [0,1]:
                self.digitsCenter = sorted(self.digitsCenter, key=lambda x: int(x[m]))

                last = self.digitsCenter[0][m]

                # x, y, w, h = self.digitsCenter[0][3]
                # cv2.rectangle(self.roi, (x, y), (x + w, y + h), (255, 255, 0), 1)

                for n,i in enumerate(self.digitsCenter[1:]):
                    if (i[m] / last > 1.6):
                        ls.append(i)
                        last = i[m]
                        continue
                    last = i[m]

                    # x, y, w, h = i[3]
                    # cv2.rectangle(self.roi, (x, y), (x + w, y + h), (255, 255, 0), 1)
                    # cv2.imshow("Gyroscope-final", self.roi)
                    # cv2.waitKey(1)
                    # print('')

            for i in ls: self.digitsCenter.pop(i)
        except Exception as e:
            pass

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

    # поиск среднего угла поворота исключая выбросы
    def get_horizon(self):
        """
        1. сортировка
        2. поиск медианы
        3. удаление выбросов
        4. нахождение средней
        :return:
        """
        def reject_outliers(data, m=2.):
            # if not isinstance(data, np.ndarray):
            data = np.array(data)
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / mdev if mdev else 1.
            res = data[s < m]
            if type(res[0]) != np.float64:
                return res[0]
            return res
            # else:
            #     d = np.abs(data - np.median(data))
            #     mdev = np.median(d)
            #     s = d / mdev if mdev else 0.
            #     return data[s < m]

        # TODO можно считать по 2 цифрам, если их наклон будет совпадать
        # TODO но для этого надо считать разницу их наклона между собой

        # обновление горизонтали только при нахождении не менее 3 цифр
        # if not (2 < len(self.angels) < 7):
        # aa = len(self.digitsCenter)
        if  1 > len(self.digitsCenter):
            return
        else:
            self.get_timer_center()
            self.horizon = np.average(reject_outliers(self.angels), axis=0)

            # абсолютный центр таймера
            x = self.roiPos[0][0] + self.timerRoiCenter[0]
            y = self.roiPos[0][1] + self.timerRoiCenter[1]

            # смещение таймера в кадре
            try:
                x1 = self.timerAbsCenter[0] - x
                y1 = self.timerAbsCenter[1] - y
                self.roiPos[0][0] -= x1
                self.roiPos[0][1] -= y1
                self.roiPos[1][0] -= x1
                self.roiPos[1][1] -= y1
            except:
                pass

            self.timerAbsCenter = (x,y)

            # cv2.circle(self.img, self.timerAbsCenter, 5, (0, 255, 255), -1)
            # cv2.imshow("Gyroscope1", self.img)
            # cv2.waitKey(1)
            # print('')

    # пока fixed можно не использовать
    def avg_horizon_trace(self):
        """
        look Sensor.avg_color_trace()
        :return:
        """
        counts = []
        for n in range(1, self.buffer.maxsize + 1):
            counts.append(1)  # fixed
            # counts.append(round(self.reaction/((n+1)))) # parabola
            # counts.append(round((n + 1) / self.reaction))  # linier reverse
        # counts = counts[::-1]

        frames = list(self.buffer.queue)
        ls = []
        for n, i in enumerate(counts):
            # print(n,i)
            if i == 0: break
            while counts[n] != 0:
                ls.append(frames[n])
                counts[n] -= 1

        avg = np.average(ls, axis=0)
        return avg

    def get_center(self, x, y, w, h):
        return (round(w / 2 + x), round(h / 2 + y))

    def get_timer_center(self):
        '''
        Самый верный способ найти точку схождения дорог - поиск
        середины всех цифр, т.к. она всегда находятся под ними.
        '''
        # TODO делать смещения если найдены только крайние цифры
        # self.digitsCenter = sorted(self.digitsCenter)

        x = np.average([i[0] for i in self.digitsCenter], axis=0)
        y = np.average([i[1] for i in self.digitsCenter], axis=0)

        # y = np.average([self.digitsCenter[0][1],self.digitsCenter[-1][1]], axis=0)
        # self.digitsCenter = sorted(self.digitsCenter, key=lambda x: int(x[0]))
        # x = np.average([self.digitsCenter[0][0],self.digitsCenter[-1][0]], axis = 0)

        # оставить прежнее значение
        try:
            self.timerRoiCenter = (int(x), int(y))
        except:
            pass

    def update(self, img):
        self.img = img
        self.roi = self.get_roi()
        self.get_angels()
        # после нахождения цифр есть
        # их углы наклона и точки центров
        # self.update_roi_pos()
        self.get_horizon()
        self.buffer.get()
        self.buffer.task_done()
        self.buffer.put(self.horizon)

        # отладка
        # print('!#roiPos',self.roiPos)

        return np.average(list(self.buffer.queue), axis=0)

if __name__ is '__main__':
    raise Exception('Use xBot to start')
else:
    from __main__ import np, cv2, math, Queue, color_rgb_filter