########################
# Starting point of processing frame.
# Get information about screenshots. Cutting frames by zones and assign sensors.
# Then passes newest zones to sensors to update their.
# Another function is searching the finish target - the place of plane destination.
# When agent know where finish and plane, he can looking for blocks on his way.
# It must be after Gyroscope because this place always placed under the timer.
########################

import time


class Vision:
    # не наследуется
    # __slots__ = ['imgHeight', 'imgWidth', 'sensors', 'yStep',
    #         'xStep', 'roiCoordinats', 'roiList', 'roiSensors']

    def __init__(self, img, roi_pos, settings, gyroscope):
        print('Vision initialization')
        self.gyroscope = gyroscope # ссылка на гироскоп
        self.imgHeight = img.shape[0]
        self.imgWidth = img.shape[1]
        self.yStep = self.imgHeight // settings['sensors']['gridRows']
        self.xStep = self.imgWidth // settings['sensors']['gridColums']
        self.imgHeight -= settings['sensors']['gridRows']
        self.imgWidth -= settings['sensors']['gridColums']
        self.finish_dist = img.shape[0] * 0.08  # по сути пересчитывается
        self.finish_point = None
        self.agentArea = [int(img.shape[0] * img.shape[1] / 500), int(img.shape[0] * img.shape[1] / 30)]
        self.agentRoiTop = int(self.imgHeight / 1.9)    # вверхняя граница зоны захвата
        self.agentRoiBott = int(self.imgHeight * 0.9)    # нижняя граница зоны захвата
        self.agentPos = None
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
        # индекс roi для сенсора
        self.cut_img(img)
        self.roiSensors = roi_pos
        print('Sensors initialization')
        self.sensors = {}
        for i in self.roiSensors:
            sensor = Sensor(self.roiList[i], settings)
            self.sensors[i] = sensor

        # self.inGame_sensors(img)
        self.plane = {'sensors': {'front' : Sensor(self.roiList[i], settings),
                                  'back': Sensor(self.roiList[i], settings),
                                  'left': Sensor(self.roiList[i], settings),
                                  'right': Sensor(self.roiList[i], settings)
                                  },
                      'top': [],
                      'bottom': []}
        # self.templates = {}
        # self.create_templates()

    def create_templates(self):
        # запоминает агента
        self.templates['plane'] = []
        dir = './templates/plane/'
        try:
            for i in os.listdir(dir):
                if not i.endswith('.png'): continue

                # tmpl = cv2.imread(dir + i, cv2.IMREAD_GRAYSCALE)
                tmpl = cv2.imread(dir + i, cv2.IMREAD_UNCHANGED)
                tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2BGRA)
                tmpl = color_rgb_filter(tmpl, 200)
                tmpl = cv2.GaussianBlur(tmpl, (3, 3), 0)
                edged = cv2.Canny(tmpl, 10, 250)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

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
            # monitor.show_result(roi)
            # time.sleep(0.5)
            # [[imgZone], (stPx), (endPx), (centerPos)]
            self.roiList.append([roi, (pos[0][0], pos[0][1]), (pos[1][0], pos[1][1])])

        # monitor.show_result(img)

    def inGame_sensors(self, img):
        ls = [[(0.12,0.06),(0.17,0.1)]]
        for n,px in enumerate(ls):
            x = int(self.imgHeight * px[0][0])
            y = int(self.imgWidth * px[0][1])
            x1 = int(self.imgHeight * px[1][0])
            y1 = int(self.imgWidth * px[1][1])

            cv2.rectangle(img, (y, x),(y1,x1), (255, 255, 255), 1)
            roi = img[x:x1, y:y1]
            self.sensors[n] = roi

        cv2.imshow("img", img)
            # cv2.imshow("img2", roi)
        cv2.waitKey(0)
        input('')

        # cv2.rectangle(img, (pos[0][0], pos[0][1]), (pos[1][0], pos[1][1]), (0, 255, 0))

    def cut_img_deb(self, img, all=True):
        if all:
            ls = self.roiCoordinats
        else:
            ls = self.roiSensors
        
        self.roiList.clear()
        for pos in ls:
            cv2.rectangle(img, (pos[0][0], pos[0][1]), (pos[1][0], pos[1][1]), (0, 255, 0))
        monitor.show_result(img)
        
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
        avgDigitH = [self.gyroscope.digits[i]['height'] for i in self.gyroscope.digits]
        #TODO чем больше цифра - тем меньше расстояние
        dist = self.finish_dist - np.average(avgDigitH, axis=0) * -0.1
        # np.average([self.digits[i]['center'] for i in self.digits], axis=1).tolist()
        x, y = self.gyroscope.timerAbsCenter

        # смещение точки по наклону таймера; вращаем вокруг x,y
        try:
            y1 = int(y + dist)
            # self.finish_point = (x, y1)
            x2 = (y1 - y) * math.sin(math.radians(-self.gyroscope.horizon)) + x
            y2 = (y1 - y) * math.cos(math.radians(-self.gyroscope.horizon)) + y
            self.finish_point = (int(x2), int(y2))
        except:
            # self.finish_point = (x, y)
            pass

        # точка схождения дорог
        # точка схождения дорог с учётом наклона
        # cv2.circle(img, self.gyroscope.timerAbsCenter, 5, (0, 255, 180), -1)
        # cv2.circle(img, self.finish_point, 5, (0, 255, 0), -1)

    def find_plane(self, img):
        src = img[self.agentRoiTop : self.agentRoiBott, 0 : self.imgWidth]
        roi = color_rgb_filter(src, 200)
        # cv2.imshow("Vision2", roi)
        # cv2.waitKey(1)
        # roi = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi, 100, 200)

        # группировка близких точек
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # dilated = cv2.dilate(edges, kernel)

        # use _,cnts,_ for opencv3 versions
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:

            # for tmpl in self.templates['plane']:
            #     result = cv2.matchTemplate(roi, tmpl[0], cv2.TM_CCOEFF_NORMED)
            #     loc = np.where(result >= 0.4)
            #     for pt in zip(*loc[::-1]):
            #         cv2.rectangle(src, pt, (pt[0] + tmpl[1], pt[1] + tmpl[2]), (0, 255, 0), 3)
            #         cv2.imshow("img", src)
            #         cv2.waitKey(0)


            rect = cv2.minAreaRect(cnt)
            area = int(rect[1][0] * rect[1][1])
            approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)

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
            if ratio > 4: continue

            # a = len(approx)
            # for i in approx:
            #     print(i)
            #     cv2.circle(src, tuple(i[0]), 3, (0, 255, 0), 1)
            #     cv2.imshow("Vision", src)
            #     cv2.waitKey(1)
            #     print('')

            # фильтр по верщинам и выхода за заону
            if 12 < len(approx) < 5: continue
            try:
                # self.plane['top'] = approx[0][0]
                # self.plane['bottom'] = approx[0][0]
                for px in approx:
                    assert px[0][1] > 2
                    # if px[0][1] < self.plane['top'][1]:
                    #     self.plane['top'] = px[0]
                    # elif px[0][1] > self.plane['bottom'][1]:
                    #     self.plane['bottom'] = px[0]
            except Exception as e:
                continue

            # self.plane['bottom'][1] += self.agentRoiTop
            # self.plane['top'][1] += self.agentRoiTop

            # нахождение центра верхней строны бокса. От него отклвдывается сенсор перед самолётом
            x, y, w, h = cv2.boundingRect(cnt)
            center = self.gyroscope.get_center(x, y, w, h)
            center = (center[0], center[1] + self.agentRoiTop)

            self.plane['cnt'] = cnt
            self.plane['approx'] = approx
            self.plane['center'] = center

            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.drawContours(src, approx, -1, (0, 0, 0), 3)
            cv2.circle(img, self.plane['center'], 3, (0, 0, 0), 1)
            # cv2.circle(img, tuple(self.plane['top']), 5, (0, 150, 150), 1)
            # cv2.circle(img, tuple(self.plane['bottom']), 5, (150, 0, 150), 1)
            # cv2.imshow("Vision", src)
            # cv2.waitKey(1)
            # print('')
            break

    def roads_marker(self, img):
        src = img[self.agentRoiTop: self.agentRoiBott, 0: self.imgWidth]

    def plane_sensors(self,img):
        def update_sensor(target, point):
            try:
                r = 20
                roi = img[point[1] - r:int(point[1] + 2 * r / 2),
                          point[0] - r:int(point[0] + 2 * r / 2)]
                if len(roi[0]) == 0:
                    # print('pass')
                    return
                # cv2.imshow("Vision2", roi)
                # cv2.waitKey(1)
                self.plane['sensors'][target].update([roi,0])
                cv2.circle(img, point, 9, self.plane['sensors'][target].avgColorTrace, -1)
                cv2.circle(img, point, 10, (255, 255, 255), 1)

                for i in safety:
                    if self.plane['sensors'][target].colorName in safety[i]:
                        self.plane['sensors'][target].safety = i
                        return

            except Exception as e:
                pass
                # print(f'plane sensor update {e}')

        # TODO find better positions
        # определения центра сенсоров
        try:
            pw, ph = self.plane['center']
        except:
            return
        fw, fh  = self.finish_point

        # передний на пол пути до точки финиша
        h = ph - (ph - fh) * 0.6
        if pw < fw:
            w = pw + (fw - pw) * 0.6
        else:
            w = pw - (pw - fw) * 0.6
        front = (int(w), int(h))
        update_sensor('front', front)

        h = ph - (ph - fh) * -0.5
        if pw < fw:
            w = pw + (fw - pw) * -0.7
        else:
            w = pw - (pw - fw) * -0.7
        back = (int(w), int(h))
        update_sensor('back', back)

        # c = 90 * (w/h)
        # x2 = (fh - ph) * math.sin(math.radians(c)) + pw * 0.6
        # y2 = (fh - ph) * math.cos(math.radians(c)) + ph
        # left = (int(x2), int(y2))
        # update_sensor('left',left)
        #
        # d = 180 + c
        # x2 = (fh - ph) * math.sin(math.radians(d)) + pw * 1.4
        # y2 = (fh - ph) * math.cos(math.radians(d)) + ph
        # right = (int(x2), int(y2))
        # update_sensor('right',right)

        # left = (int(self.imgWidth * 0.15), self.plane['center'][1] - 10)
        left = [int(self.imgWidth * 0.24), front[1] + 15]
        if self.plane['center'][0] - left[0] < 50:
            # оттодвигает сенсор чтобы он не попадал на самолёт.
            left[0] -= int(left[0]/self.plane['center'][0]*40)
            left[1] -= int(self.plane['center'][0]/left[0]*30)
        # elif self.plane['center'][0] - left[0] > 150:
        #     a = self.plane['center'][0] - left[0]
        #     left[0] += int(a / 2)
        #     left[1] += int(self.plane['center'][0]/left[0]*30)
        update_sensor('left',tuple(left))

        right = [int(self.imgWidth * 0.76), front[1] + 15]
        aa = right[0] / self.plane['center'][0]
        if right[0] - self.plane['center'][0] < 50:
            right[0] += int(self.plane['center'][0]/right[0]*40)
            right[1] -= int(self.plane['center'][0]/right[0]*30)
        # elif right[0]/self.plane['center'][0] > 1.8:
        #     a = right[0] - self.plane['center'][0]
            # right[0] -= int(a / 2)
            # right[1] += int(self.plane['center'][0]/right[0]*30)
        update_sensor('right',tuple(right))

        # cv2.line(img, self.plane['center'],self.plane['sensors']['front'], (0, 0, 0), 1)

    def look(self, img):
        self.cut_img(img, all=True)

        try:
            self.find_finish_point(img)
        except Exception as e:
            print(f'Vision: cannot find finish point: {e}')
        self.update_sensors()
        self.find_plane(img)
        try:
            self.plane_sensors(img)
        except Exception as e:
            pass
            # print(f"Vision: can't get plane sensors: {e}")
        return self.sensors

if __name__ == '__main__':
    raise Exception('Use xBot to start')
else:
     from __main__ import os, cv2, np, math, Sensor, color_rgb_filter, dist, safety, monitor