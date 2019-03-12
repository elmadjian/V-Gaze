import evdev
from evdev import InputDevice, categorize, ecodes

def run():
    '''
    Callback for keyboard
    's' KEY is used to toggle pupil action area mapping
    'r' KEY resets the pupil action model
    'c' KEY is used to trigger calibration procedure
    'n' KEY is used to make calibration targets move to the next position
    'q' KEY is used to quit
    'SHIFT + [1-9]' selects a calibration
    'SHIFT + 0' disables calibration 
    '''
    device = None
    devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
    for d in devices:
        print(d)
        # if "keyboard" in d.name or 'Keyboard' in d.name:
        #     device = evdev.InputDevice('/dev/input/event3')
        #     break

    calibrations = [i for i in range(2,12)]
    device = evdev.InputDevice('/dev/input/event26')

    if device is not None:
        print(device.name)
        for event in device.read_loop():
            print(event)
            # if event.type == ecodes.EV_KEY:
            #     print(event.code)

if __name__=='__main__':
    run()