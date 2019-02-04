########################
# Agent logic and keys controller
########################

import win32con
import win32api
import time

class Controller:
    def __init__(self):
        print('Controller initialization')
        from pynput.keyboard import Key, Controller


        self.keyboard = Controller()
        self.key = Key
        # self.keyboard.press(Key.left)
        # self.keyboard.press(Key.right)
        self.lastPress = time.time()

    def play(self, planeSens):
        # print(time.time())
        if planeSens['front'].safety > 2 and planeSens['back'].safety > 2:
            if planeSens['left'].safety < planeSens['right'].safety:
                # if time.time() - self.lastPress > 0.5:
                self.keyboard.press(self.key.left)
                self.keyboard.release(self.key.left)
                self.lastPress = time.time()
                print('go left')
            elif planeSens['left'].safety > planeSens['right'].safety:
                # if time.time() - self.lastPress > 0.5:
                self.keyboard.press(self.key.right)
                self.keyboard.release(self.key.right)
                self.lastPress = time.time()
                print('go right')
            else:
                if planeSens['left'].safety < 4:
                    # if time.time() - self.lastPress > 0.5:
                    self.keyboard.press(self.key.left)
                    self.keyboard.release(self.key.left)
                    self.lastPress = time.time()
                    print('go left')
                else:
                    print('wait')
        else:
            pass
            # print('stay')


