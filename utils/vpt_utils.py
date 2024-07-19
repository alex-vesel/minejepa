

ACTION_MAP = {
    'ESC':          None,
    'attack':       None,
    'back':         "key.keyboard.s",
    'camera_pitch': None,
    'camera_yaw':   None,
    'drop':         "key.keyboard.q",
    'forward':      "key.keyboard.w",
    'hotbar_1':     "key.keyboard.1",
    'hotbar_2':     "key.keyboard.2",
    'hotbar_3':     "key.keyboard.3",
    'hotbar_4':     "key.keyboard.4",
    'hotbar_5':     "key.keyboard.5",
    'hotbar_6':     "key.keyboard.6",
    'hotbar_7':     "key.keyboard.7",
    'hotbar_8':     "key.keyboard.8",
    'hotbar_9':     "key.keyboard.9",
    'inventory':    "key.keyboard.e",
    'jump':         'key.keyboard.space',
    'left':         'key.keyboard.a',
    'pickitem':     None,
    'right':        'key.keyboard.d',
    'sneak':        'key.keyboard.left.shift',
    'sprint':       'key.keyboard.left.control',
    'swapHands':    None,
    'use':          None,
    'f3':           None,
}


def vpt_dpixels_to_degrees(d_pixels):
    # conversion of pixels to degrees for OpenAI VPT
    # FOV = 70 degrees, GUI scale is 2
    return d_pixels * 0.15