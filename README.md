# Sets with friends solver
Automated solver for sets with friends! Original code skeleton and functions adapted from AceLewis.

Environment that works for me:
- Microsoft Edge browser
- 80% magnification

## solver.py
Original code file adapted from AceLewis' original solution from https://github.com/AceLewis/set-solving-bot, works with the android version of sets with friends

## hxsolver.py
Code file that works with the browser version of Sets with friends https://setwithfriends.com/, run the code and it solves the entire board for you. Make sure that your browser with setswithfriends is on the prev tab (ie alt-tab will bring you to the browser with sets with friends).

## pranksolver.py
Code file that also works with the browser version of Sets with friends. Only clicks on sets upon pressing the spacebar.

## Adjustable variables
- SET_BOUNDS (boundary of the game screenshot, ensure that a screenshot of the entire game board is taken)
- BOX_WIDTH (width of each card box)
- BOX_HEIGHT (height of each card box)
- TOP_LEFT_BOX (tuple of the bounding boxes of the first card box on the top left. left, top, right, bottom)
- MAX_CARDS (maximum number of cards that can be displayed on the board)
- CLICK_DELAY (delay between the 3 clicks in each turn; 0.02 should work)
- ROUND_DELAY (delay between clicking 2 separate sets inside hxsolver.py)
- STARTING_DELAY (delay at the start before the script starts clicking for hxsolver.py)
- size_modifier (size modifier constant for area estimations for shape estimation)
- APPROX_VOLUME (modify the size_modifier attribute, try not to edit this)
- APPROX_FILL (percentage fill of each shape)
- COLOUR_DICT (colour dictionary, use a browser eyedropper to determine the colour of the shapes for your browser, but should be correct)
