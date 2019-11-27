import pyautogui
import cv2
from scipy.spatial import distance

def distance_derivative(previous_dist,current_dist,fps):
    return fps*(current_dist-previous_dist)

def rotate(middle, data_coords_open, data_coords_closed, opened, closed, im0):
    middle_x, middle_y = middle
    anchor_x, anchor_y = data_coords_open[0][2:]
    x_change = opened['mouse_x'] - anchor_x
    y_change = opened['mouse_y'] - anchor_y
    pyautogui.moveTo(middle_x+x_change, middle_y+y_change/2)
    pyautogui.keyDown('shift')
    pyautogui.mouseDown(button='middle')
    pyautogui.keyUp('shift')
    im0 = cv2.line(im0,data_coords_closed[-1][:2],data_coords_open[-1][:2],(255,0,255),2)
    return im0
def zoom(hand_a, hand_b, opened_distance, fps, im0):
    im0 = cv2.line(im0,(hand_a["center_x"],hand_a["center_y"]),(hand_b["center_x"],hand_b["center_y"]),(0,255,0),2)
    dist = distance.euclidean((hand_a["center_x"],hand_a["center_y"]), (hand_b["center_x"],hand_b["center_y"]))
    opened_distance.append(dist)
    dydx = 0
    if len(opened_distance) > 1:
        previous_distance = opened_distance[-2]
        dydx = distance_derivative(previous_distance,dist,fps)
        im0 = cv2.putText(im0, "dy/dx:{:0.2f}".format(dydx), (10,460), cv2.FONT_HERSHEY_SIMPLEX , 1,(0, 255, 0) , 2, cv2.LINE_AA)
    return im0, opened_distance, dydx
def pan(coords):
    x,y = coords
    pyautogui.mouseDown(button='middle')
    pyautogui.moveTo(x, y)
def mouse_move(coords):
    x,y = coords
    pyautogui.moveTo(x, y)
