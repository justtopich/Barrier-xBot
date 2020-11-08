########################
# Identifying shades of colors.
# This class reduce count of values by grouping the similar colors and return human titles.
# It's work faster then convert RGB to HSV with color range.
# Almost any way I need grouping colors for agent logic.
#######################



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

        a = (min_colours[min(min_colours.keys())])
        if a == 'tomato':
            a = min(min_colours.keys())

        return min_colours[min(min_colours.keys())]

    def get_html_name(self, rgbIn):
        rgb = (rgbIn[2], rgbIn[1], rgbIn[0])
        try:
            name = webcolors.rgb_to_name(rgb)
        except ValueError:
            name = self.closest_name(rgb)
        return name

if __name__ is '__main__':
    raise Exception('Use xBot to start')
else:
    from __main__ import np, cv2, webcolors, dist