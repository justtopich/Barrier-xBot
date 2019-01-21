########################
# Sensors used to get information about color in specific zone - ROI. Zones are formed in class Vision.
# Each sensor has its own ROI. The main function is calculating average color.
# They have inertia and reaction of results changes. It allow to cover some inaccuracies in processing.
########################


class Sensor:
    __slots__ = ('img', 'startPx',
                 'endPx', 'centerPx',
                 'reaction', 'avgColor',
                 'avgColorTrace',
                 'colorName', 'buffer')

    def __init__(self, roi, settings):
        self.img = roi[0]
        self.startPx = roi[1]
        self.endPx = roi[2]
        self.centerPx = self.get_center()
        # self.lastState = None
        self.reaction = settings['sensors']['reaction']
        self.avgColor = self.avg_color()
        self.avgColorTrace = self.avgColor
        self.colorName = self.avg_color_name(self.avgColor)
        self.buffer = Queue()
        self.buffer.maxsize = settings['sensors']['bufferSize']
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
        for n in range(1, self.buffer.maxsize + 1):
            # counts.append(1) # fixed
            # counts.append(round(self.reaction/((n+1)))) # parabola
            counts.append(round((n + 1) / self.reaction))  # linier reverse
        counts = counts[::-1]

        frames = list(self.buffer.queue)
        ls = []
        for n, i in enumerate(counts):
            # print(n,i)
            if i == 0: break
            while counts[n] != 0:
                ls.append(frames[n])
                counts[n] -= 1
        # input()
        if len(ls) == 0:
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

if __name__ is '__main__':
    raise Exception('Use xBot to start')
else:
    from __main__ import np, Queue
    from ColorLabeler import ColorLabeler

    colorLabeler = ColorLabeler()