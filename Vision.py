########################
# Starting point of processing frame.
# Get information about screenshots. Cutting frames by zones and assign sensors.
# Then passes newest zones to sensors to update their.
########################


class Vision:
    # не наследуется
    # __slots__ = ['imgHeight', 'imgWidth', 'sensors', 'yStep',
    #         'xStep', 'roiCoordinats', 'roiList', 'roiSensors']

    def __init__(self, img, roi_pos, settings, gyroscope):
        self.gyroscope = gyroscope # ссылка на гироскоп
        self.imgHeight = img.shape[0]
        self.imgWidth = img.shape[1]
        self.yStep = self.imgHeight // settings['sensors']['gridRows']
        self.xStep = self.imgWidth // settings['sensors']['gridColums']
        self.imgHeight -= settings['sensors']['gridRows']
        self.imgWidth -= settings['sensors']['gridColums']
        self.finish_dist = img.shape[0] * 0.08
        self.finish_point = None
        self.agentArea = [int(img.shape[0] * img.shape[1] / 500), int(img.shape[0] * img.shape[1] / 40)]


        # координаты roi
        self.roiCoordinats = []
        for y in range(0, self.imgHeight, self.yStep):
            for x in range(0, self.imgWidth, self.xStep):
                y1 = y + self.yStep
                x1 = x + self.xStep
                # cell = img[y:y + self.yStep, x:x + self.xStep]
                self.roiCoordinats.append([[x, y], [x1, y1]])

        # весь кадр по частям
        self.roiList = []
        self.templates = {}
        # индекс roi для сенсора
        self.cut_img(img)
        self.roiSensors = [(round(len(self.roiList) * n)) for n in roi_pos]
        self.sensors = {}
        for i in self.roiSensors:
            sensor = Sensor(self.roiList[i], settings)
            self.sensors[i] = sensor

    def create_templates(self):
        # запоминает агента
        self.templates['plane'] = []
        dir = './templates/plane/'
        try:
            for i in os.listdir(dir):
                if not i.endswith('.png'): continue

                tmpl = cv2.imread(dir + i, cv2.IMREAD_GRAYSCALE)
                # img = cv2.imread(dir+i)
                # tmpl = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                # tmpl = color_rgb_filter(tmpl)
                # tmpl = cv2.Canny(tmpl, 50, 200)
                h, w = tmpl.shape[:2]
                self.templates['plane'].append([tmpl, w, h])
                # break
        except Exception as e:
            raise Exception(f'Fail to open templates: {e}')

    # возращает весь кадр по частям
    def cut_img(self, img, all=True):
        if all is False:
            ls = self.roiSensors
        else:
            ls = self.roiCoordinats
        self.roiList.clear()
        for pos in ls:
            # cv2.rectangle(img, (pos[0][0], pos[0][1]), (pos[1][0], pos[1][1]), (0, 255, 0))
            # cell = img[y:y + self.yStep, x:x + self.xStep]
            roi = img[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]
            # show_result(roi, do)
            # time.sleep(0.5)
            # [[imgZone], (stPx), (endPx), (centerPos)]
            self.roiList.append([roi, (pos[0][0], pos[0][1]), (pos[1][0], pos[1][1])])
        # cv2.imwrite(f"asas.png" , img)

    def cut_img_deb(self, img, all=True):
        if all is False:
            ls = self.roiSensors
        else:
            ls = self.roiCoordinats
        self.roiList.clear()
        for pos in ls:
            cv2.rectangle(img, (pos[0][0], pos[0][1]), (pos[1][0], pos[1][1]), (0, 255, 0))
            roi = img[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]
            self.roiList.append([roi, (pos[0][0], pos[0][1]), (pos[1][0], pos[1][1])])

    # получение конкретных зон и их предобработка
    def update_sensors(self):
        # print(self.sensors[15].img[0]==self.sensors[26].img[0])
        for i in self.roiSensors:
            # show_result(self.roiList[i][0], do)
            # time.sleep(0.5)
            self.sensors[i].update(self.roiList[i])

    def get_sensors(self):
        return self.sensors

    def find_finish_point(self, img):
        x, y = self.gyroscope.timerAbsCenter
        y1 = y + self.finish_dist
        self.finish_point = (x, y1)

        # смещение точки по наклону таймера
        x2 = (y1 - y) * math.sin(math.radians(-self.gyroscope.horizon)) + x
        y2 = (y1 - y) * math.cos(math.radians(-self.gyroscope.horizon)) + y
        self.finish_point = (int(x2), int(y2))

        # точка схождения дорог
        # точка схождения дорог с учётом наклона
        cv2.circle(img, self.gyroscope.timerAbsCenter, 5, (0, 255, 180), -1)
        cv2.circle(img, self.finish_point, 5, (0, 255, 0), -1)

    def find_agent(self, img):
        h = int(self.imgHeight / 2)
        src = img[h:self.imgWidth, 0:self.imgWidth]
        roi = color_rgb_filter(src)
        edges = cv2.Canny(roi, 100, 200)
        # use _,cnts,_ for old versions
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            area = int(rect[1][0] * rect[1][1])
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            box = np.int0(cv2.boxPoints(rect))

            edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
            edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
            # длина сторон
            edgeNorm = [cv2.norm(edge1), cv2.norm(edge2)]
            edgeNorm.sort()
            if edgeNorm[0] < 1: continue

            # фильтр по пропорциям
            ratio = edgeNorm[1] / edgeNorm[0]

            # if not self.agentArea[0] < area < self.agentArea[1]: continue
            if self.agentArea[0] > area: continue
            if self.agentArea[1] < area: continue
            if ratio > 5: continue
            a = len(approx)
            if len(approx) < 6: continue
            try:
                for px in approx:
                    # print(px)
                    assert px[0][1] > 2
            except:
                continue

            x, y, w, h = cv2.boundingRect(box)

            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.drawContours(src, approx, -1, (0, 255, 255), 3)
            cv2.imshow("Vision", src)
            cv2.imshow("Vision2", roi)
            cv2.waitKey(1)
            print('')

    def look(self, img):
        self.cut_img(img, all=True)
        try:
            self.find_finish_point(img)
        except Exception as e:
            pass
        self.update_sensors()
        self.find_agent(img)
        return self.sensors

if __name__ is '__main__':
    raise Exception('Use xBot to start')
else:
     from __main__ import os, cv2, np, math, Sensor, color_rgb_filter, show_result