########################
# Agent logic and keys controller
########################

import win32con
import win32api
import time

# actions
# 0 - pass
# 1 - wait
# 2 - left
# 3 - right


class Controller:
    def __init__(self):
        print('Controller initialization')
        from pynput.keyboard import Key, Controller
        self.keyboard = Controller()
        self.key = Key
        self.keyboard.press(Key.left)
        self.keyboard.release(self.key.left)
        self.keyboard.press(Key.right)
        self.keyboard.release(self.key.right)
        self.lastPress = time.time()
        self.action = 0
        self.waitCount = 0
        self.maxWait = 20
        self.pressWait = 0.18
        # self.pressWait = 0.25

    def go_left(self):
        if self.action == 2 and time.time() - self.lastPress > self.pressWait:
            self.keyboard.press(self.key.left)
            self.keyboard.release(self.key.left)
            self.lastPress = time.time()
            print('go left')
        else:
            self.action = 2

    def go_right(self):
        if self.action == 3 and time.time() - self.lastPress > self.pressWait:
            self.keyboard.press(self.key.right)
            self.keyboard.release(self.key.right)
            self.lastPress = time.time()
            print('go right')
        else:
            self.action = 3

    def play(self, planeSens):
        # print(time.time())
        if planeSens['front'].safety > 3 and planeSens['back'].safety > 3:
            print(planeSens['left'].safety, planeSens['right'].safety)
            if planeSens['left'].safety < 4 or planeSens['right'].safety < 4:
                if planeSens['left'].safety < planeSens['right'].safety:
                    self.go_left()
                    self.waitCount = 0
                elif planeSens['left'].safety > planeSens['right'].safety:
                    self.go_right()
                    self.waitCount = 0
                else:
                    self.go_right()
                    self.waitCount = 0

            else:
                print('wait', self.waitCount)
                self.waitCount += 1
                if planeSens['front'].safety + planeSens['back'].safety > 11 or self.waitCount > self.maxWait:
                    self.go_right()
                    self.do = 3
                    self.waitCount = 0
                elif planeSens['left'].safety < 2:
                    self.go_left()
                    self.do = 2
                    self.waitCount = 0
                else:
                    self.action = 1
        else:
            self.action = 0