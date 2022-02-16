from mss import mss
from PIL import Image
import time
import math
import numpy as np
import cv2
import itertools
import win32api
import win32con

# Adjustable variables
SET_BOUNDS = {"left":600,"top":200,"width":800,"height":900} # bounds of the game screenshot
BOX_WIDTH = 200 # width of each card box
BOX_HEIGHT = 115 # height of each card box
TOP_LEFT_BOX = (55, 40, 55+BOX_WIDTH, 40+BOX_HEIGHT) # top left card box
MAX_CARDS = 15 # max number of cards
CLICK_DELAY = 0.02 # delay between the 3 clicks in each turn
ROUND_DELAY = 2 # delay between round clicks
STARTING_DELAY = 5 # delay at the start

size_modifier = 0.89 # size modifier for area estimations
APPROX_VOLUME = {2500*size_modifier: 'diamond', # don't change this
                3300*size_modifier: 'squiggle',
                4300*size_modifier: 'oval'}

APPROX_FILL = {0: 'solid', 0.6: 'open', 0.2: 'striped'} # percentage fill of each shape

COLOUR_DICT = {(128, 0, 128): 'purple', # colour dictionary
                   (255, 6, 7): 'red',
                   (0, 128, 13): 'green'}

def get_cards(box_dict):
    """Get information about all the cards"""
    cards = []
    for box in box_dict.values():
        if not is_it_a_card(box):
            # All cards found
            break
        card = get_card(box)
        if card:   
            cards.append(get_card(box))
    return cards

def get_card(box):
    """Find details about the card"""
    grey_image = np.array(cv2.cvtColor(np.float32(box), cv2.COLOR_RGB2GRAY),
                          dtype='uint8')
    # Find the colour from the darkest pixel
    _, _, min_loc, _ = cv2.minMaxLoc(grey_image)
    colour_pixel = np.float32(box)[min_loc[1]][min_loc[0]][:3]
    colour_obtained = find_closest_colour(colour_pixel)
    # Threshold the image so white-ish pixels are seperated from all other
    _, thresh = cv2.threshold(grey_image, 240, 255, cv2.THRESH_BINARY_INV)
    # Get background
    background = isolate_background(thresh)
    # Look at strip in background to find the number of shapes from the number
    # of times it changes from black to white
    half_card_strip = background[int(background.shape[0]/2), :]
    number_of_shapes = int(sum(np.roll(half_card_strip, 1) != half_card_strip)/2)
    if not number_of_shapes:
        return None
    # Calculate the volume per shape and infer what shape it is
    volume_per_shape = np.sum(background == 255)/number_of_shapes
    shape = find_shape(volume_per_shape)
    # Find the volume inside the shape that is white and infer the fill
    volume_inside_per_shape = np.sum((background-thresh) == 255)/number_of_shapes
    percent_filled = volume_inside_per_shape/volume_per_shape
    fill = find_fill(percent_filled)

    return (number_of_shapes, colour_obtained, fill, shape)


def shift_roi(roi, iteration):
    shift_right = 240
    shift_down = 148
    left, upper, right, bottom = roi
    new_tuple = (left+(iteration%3)*shift_right, upper+(iteration//3)*shift_down, right+(iteration%3)*shift_right, bottom+(iteration//3)*shift_down)
    return new_tuple

def is_white(pixel):
    """Check if a pixel is white-ish"""
    return all(x > 240 for x in pixel)

def is_it_a_card(box):
    """Check if the image is a card by seeing if the top left pixel and
    bottom right are white."""
    box_size = box.size
    bottom_right = (box_size[0]-1, box_size[1]-1)
    bottom_left = (0,box_size[1]-1)
    return is_white(box.getpixel(bottom_left)) and is_white(box.getpixel(bottom_right))

def is_it_a_deck(box_dict):
    """Check if a deck is on screen"""
    # It is a deck if there are a number of cards divisible by three
    # and the cards are in order. e.g 3 cards are not selected as
    # green but not matching
    cards_found = [is_it_a_card(box) for box in box_dict.values()]
    num_of_cards = sum(cards_found)
    is_a_deck = num_of_cards % 3 == 0 and all(cards_found[0:num_of_cards])
    return num_of_cards*is_a_deck

def isolate_background(image):
    """Function to isolate the background, assumes the background is white"""
    # Copy the thresholded image.
    floodfill_image = image.copy()

    h, w = floodfill_image.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(floodfill_image, mask, (0, 0), 255)
    # Get background from these two images
    return image | cv2.bitwise_not(floodfill_image)

def find_shape(num):
    """Find the shape from the volume"""
    return APPROX_VOLUME[min(APPROX_VOLUME.keys(), key=lambda x: abs(x-num))]

def distance(colour_1, colour_2):
    """Find the 'distance' between two colours"""
    return math.sqrt(sum((x-y)**2 for x, y in zip(colour_1, colour_2)))

def find_closest_colour(input_colour):
    """Find the closest colour in the dictionary"""
    colours = list(COLOUR_DICT.keys())
    closest_colour = sorted(colours,
                            key=lambda colour: distance(colour, input_colour))[0]
    return COLOUR_DICT[closest_colour]

def find_fill(fill_percentage):
    """Find the shape from the volume"""
    return APPROX_FILL[min(APPROX_FILL.keys(), key=lambda x: abs(x-fill_percentage))]

def find_missing_card(card_1, card_2):
    """Find the missing card in the set"""
    # Make a set for all possibilities
    number_set = set(range(1, 4))
    colour_set = set(COLOUR_DICT.values())
    filled_set = set(APPROX_FILL.values())
    shape_set = set(APPROX_VOLUME.values())

    # Make set for both cards
    number_set_cards = set([card_1[0], card_2[0]])
    colour_set_cards = set([card_1[1], card_2[1]])
    filled_set_cards = set([card_1[2], card_2[2]])
    shape_set_cards = set([card_1[3], card_2[3]])

    # Find the smallest set;
    # If they are the same the smallest is the set of the cards.
    # If they are different then the smallest set is all possibilities minus the current set.
    matching_num = next(iter(min((number_set-number_set_cards, number_set_cards), key=len)))
    matching_colour = next(iter(min((colour_set-colour_set_cards, colour_set_cards), key=len)))
    matching_filled = next(iter(min((filled_set-filled_set_cards, filled_set_cards), key=len)))
    matching_shape = next(iter(min((shape_set-shape_set_cards, shape_set_cards), key=len)))

    return (matching_num, matching_colour, matching_filled, matching_shape)

def click_location(position):
    top_left_centre_x = 755
    top_left_centre_y = 300
    shift_right = 240
    shift_down = 148

    top_left_centre_x += (position%3)*shift_right
    top_left_centre_y += (position//3)*shift_down

    win32api.SetCursorPos((top_left_centre_x, top_left_centre_y))
    time.sleep(CLICK_DELAY)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, top_left_centre_x, top_left_centre_y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, top_left_centre_x, top_left_centre_y, 0, 0)


def screenshot():
    with mss() as sct:
        scr = sct.grab(SET_BOUNDS)
        return Image.frombytes('RGB', scr.size, scr.bgra,'raw', 'BGRX')

def test_step():
    screenie = screenshot()
    box_dict = {}
    # for i in range(MAX_CARDS):
    #     card = screenie.crop(box=shift_roi(TOP_LEFT_BOX, i))
    #     box_dict[i] = card
    #     card.save(f'{i}.png')

    box_dict = {i: screenie.crop(box=shift_roi(TOP_LEFT_BOX, i)) for i in range(MAX_CARDS)}

    cards = get_cards(box_dict)
    for card_1, card_2 in itertools.combinations(cards, 2):
        missing_card = find_missing_card(card_1, card_2)
        if missing_card in cards:
                click_location(cards.index(card_1))
                click_location(cards.index(card_2))
                click_location(cards.index(missing_card))
                break

if __name__ == '__main__':
    prevPressed = False
    time.sleep(0.5)
    while True:
        if win32api.GetAsyncKeyState(32) and not prevPressed:
            prevPressed = True
            test_step()
        elif not win32api.GetAsyncKeyState(32) and prevPressed:
            prevPressed = False