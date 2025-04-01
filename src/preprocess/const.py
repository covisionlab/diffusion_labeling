# label encoding specification
ID_to_NAME_and_COLOR = {
    1: ('live_knot', (0, 255, 0)),          # 1
    2: ('dead_knot', (255, 0, 0)),          # 1
    3: ('knot_missing', (255, 100, 0)),     # 1
    4: ('knot_with_crack', (255, 175, 0)),  # 1
    5: ('crack', (255, 0, 100)),            # 2
    6: ('quartzity', (100, 0, 100)),        # 3
    7: ('resin', (255, 0, 255)),            # 4
    8: ('marrow', (0, 0, 255)),             # 5
    9: ('blue_stain', (16, 255, 255)),      # 6
    10: ('overgrown', (0, 64, 0))
}

# remapping of labels (collapsing knots i.e. 1,2,3,4 -> 1)
ID_REMAPPER = { 1:1, 2:1, 3:1, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:10 }

NAME_to_ID_and_COLOR = {
    'live_knot': ((0, 255, 0), 1),
    'dead_knot': ((255, 0, 0), 2),
    'knot_missing': ((255, 100, 0), 3),
    'knot_with_crack': ((255, 175, 0), 4),
    'crack': ((255, 0, 100), 5),
    'quartzity': ((100, 0, 100), 6),
    'resin': ((255, 0, 255), 7),
    'marrow': ((0, 0, 255), 8),
    'blue_stain': ((16, 255, 255), 9),
    'overgrown': ((0, 64, 0), 10)
}


COLOR_to_NAME_and_ID = {
    (0, 255, 0): ('live_knot', 1),
    (255, 0, 0): ('dead_knot', 2),
    (255, 100, 0): ('knot_missing', 3),
    (255, 175, 0): ('knot_with_crack', 4),
    (255, 0, 100): ('crack', 5),
    (100, 0, 100): ('quartzity', 6),
    (255, 0, 255): ('resin', 7),
    (0, 0, 255): ('marrow', 8),
    (16, 255, 255): ('blue_stain', 9),  # label 9 should not exist
    (0, 64, 0): ('overgrown', 10)
}
