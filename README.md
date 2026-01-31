# Color-Based Fruit Ninja Game (OpenCV)

This project is a webcam-based interactive game inspired by *Fruit Ninja*, implemented using **Python** and **OpenCV**. The game tracks colored objects in real time and uses their motion to slice falling fruits on the screen.


## Description

The game uses a webcam to detect **green** and **purple** colored objects in the scene. These objects act as virtual blades. Their positions are tracked over time using **Kalman filters** to smooth motion and improve robustness.

Fruits are spawned, move across the screen, and can be cut when a tracked color position intersects with a fruit.


## Features

- Real-time color-based object tracking (green & purple)
- Kalman filter for motion smoothing and prediction
- Fruit spawning and movement
- Collision detection between tracked positions and fruits
- Score, level, lives, and FPS display
- Uses PNG assets with alpha transparency

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy

## Preview

▶️ **Gameplay video preview**

[Click here to watch the demo](video/Demo_Video.mp4)

## Contributors:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/PhuongAnh2212">
        <img src="https://avatars.githubusercontent.com/u/119726597?v=4" width="100" height="100" style="border-radius: 50%;" alt="PhuongAnh2212"/>
        <br />
        <sub><b>PhuongAnh2212</b></sub>
      </a>
    </td> 
    <td align="center">
      <a href="https://github.com/thaoton1910">
        <img src="https://avatars.githubusercontent.com/u/187097297?v=4" width="100" height="100" style="border-radius: 50%;" alt="thaoton"/>
        <br />
        <sub><b>thaoton1910</b></sub>
      </a>
    </td>
  </tr>
</table>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
