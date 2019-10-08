from imports import *

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        keys.append(str(key))
        return False
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
        return False


def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def key_check():
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
        listener.join()


def keys_to_output(keys):
    
    if "'a'" in keys:
        return [1,0,0]
    elif "'d'" in keys:
        return [0,0,1]
    elif "'w'" in keys:
        return [0,1,0]
    else:
        return [0,0,0]


def PressKey(key):
    pyautogui.keyDown(key) 


def ReleaseKey(key):
    pyautogui.keyUp(key) 


def straight():
    PressKey('w')
    ReleaseKey('a')
    ReleaseKey('d')
    time.sleep(0.09)


def left():
    PressKey('a')
    PressKey('w')
    ReleaseKey('d')
    ReleaseKey('a')
    time.sleep(0.09)
    ReleaseKey('w')


def right():
    PressKey('d')
    PressKey('w')
    ReleaseKey('a')
    ReleaseKey('d')
    time.sleep(0.09)
    ReleaseKey('w')


def slow_ya_roll():
    #PressKey('s')
    ReleaseKey('w')
    ReleaseKey('a')
    ReleaseKey('d')
