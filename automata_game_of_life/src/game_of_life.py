import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw

####################################### RULES ##############################################
# Any live cell with fewer than two live neighbours dies, as if by underpopulation.
# Any live cell with two or three live neighbours lives on to the next generation.
# Any live cell with more than three live neighbours dies, as if by overpopulation.
# Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
####################################### RULES ##############################################


class GameOfLife:
    def __init__(self, length: int = 256, height: int = 521, seed: int = 42):
        np.random.seed(seed)
        self.Z = np.zeros((height, length))
        h = height // 4
        l = length // 4
        self.Z[ h : 3 * h, l : 3 * l] = np.random.randint(2, size=(2 * h, 2 * l))
        self.height = height
        self.length = length
        self.title = "gof"

    def _step(self):
        Z = self.Z
        N = (Z[:-2,:-2] + Z[:-2,1:-1] + Z[:-2,2:] +
             Z[1:-1,:-2] + Z[1:-1,1:-1] + Z[1:-1,2:] +
            Z[2:,:-2] + Z[2:,1:-1] + Z[2:,2:])

        birth = ( N == 3 ) & (Z[1:-1, 1:-1] == 0)
        survive = ((N == 3) | (N == 4)) & (Z[1:-1, 1:-1] == 1) 
        self.Z = np.zeros_like(self.Z)
        self.Z[1:-1, 1:-1][birth | survive] = 1

    def plot_final_step(self, t: int = 1):
        for _ in range(t):
            self._step()
        plt.imshow(self.Z, interpolation="nearest", cmap="Greys")
        # filename = './png/'+str('%04d' % t) + '.png'
        filename = str("%04d" % t) + ".png"
        plt.axis("off")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.clf()

    def animate_result(self, t: int = 10):
        fig, ax = plt.subplots()
        ax.axis("off")
        frames = []
        for _ in range(t):
            self._step()
            im = ax.imshow(self.Z, interpolation="nearest", cmap="Greys", animated=True)
            frames.append([im])
        anim = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
        anim.save("game_of_life_" + self.title + ".mp4", writer="ffmpeg", fps=1)
        plt.close(fig)


class Mazentic(GameOfLife):
    def __init__(self):
        super().__init__()
        self.title = "mazentic"

    def _step(self):
        Z = self.Z
        N = (Z[:-2,:-2] + Z[:-2,1:-1] + Z[:-2,2:] +
             Z[1:-1,:-2] + Z[1:-1,1:-1] + Z[1:-1,2:] +
            Z[2:,:-2] + Z[2:,1:-1] + Z[2:,2:])

        birth = ( N == 3 ) & (Z[1:-1, 1:-1] == 0)
        survive = ((N >= 3) | (N <= 6)) & (Z[1:-1, 1:-1] == 1) 
        self.Z = np.zeros_like(self.Z)
        self.Z[1:-1, 1:-1][birth | survive] = 1

class Flakes(GameOfLife):
    def __init__(self):
        super().__init__()
        self.title = "flakes"

    def _step(self):
        Z = self.Z
        N = (Z[:-2,:-2] + Z[:-2,1:-1] + Z[:-2,2:] +
             Z[1:-1,:-2] + Z[1:-1,1:-1] + Z[1:-1,2:] +
            Z[2:,:-2] + Z[2:,1:-1] + Z[2:,2:])

        birth = ( N == 3 ) & (Z[1:-1, 1:-1] == 0)
        self.Z[1:-1, 1:-1][birth] = 1
class GameOfLife_Glider(GameOfLife):
    def __init__(self, length: int = 512, height: int = 256):
        super().__init__(length=length, height=height)
        self.title = "glider"
        glider = np.array(
                    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        )

        self.Z = np.zeros_like(self.Z)

        gl_h, gl_l = glider.shape
        assert gl_l < self.length and gl_h < self.height, "Glider is bigger than grid"
        mid_start_l = self.length // 2 - gl_l // 2
        mid_end_l = mid_start_l + gl_l

        mid_start_h = self.height // 2 - gl_h // 2
        mid_end_h = mid_start_h + gl_h

        self.Z[mid_start_h:mid_end_h, mid_start_l:mid_end_l] = glider


class GameOfLife_Puffer(GameOfLife):
    def __init__(self, length: int = 512, height: int = 256):
        super().__init__(length=length, height=height)
        self.title = "puffer"
        puffer = np.array(
                    [[0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0],
                    [1,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,0,0,1],
                    [0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0],
                    [0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0],
                    [0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0]]
        )

        self.Z = np.zeros_like(self.Z)

        gl_h, gl_l = puffer.shape
        assert gl_l < self.length and gl_h < self.height, "Glider is bigger than grid"
        mid_start_l = self.length // 2 - gl_l // 2
        mid_end_l = mid_start_l + gl_l

        mid_start_h = self.height // 2 - gl_h // 2
        mid_end_h = mid_start_h + gl_h

        self.Z[mid_start_h:mid_end_h, mid_start_l:mid_end_l] = puffer

class GameOfLife_Dimond(GameOfLife):
    def __init__(self, length: int = 512, height: int = 256):
        super().__init__(length=length, height=height)
        self.title = "dimond"
        puffer = np.array(
[    [0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,1,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,1,1,0,0,1,0,0,1,1,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,1,0],
    [1,0,1,0,1,1,0,1,1,0,1,0,1],
    [0,1,0,0,0,0,1,0,0,0,0,1,0],
    [0,0,1,1,0,0,1,0,0,1,1,0,0],
    [0,0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,1,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0],
]
        )

        self.Z = np.zeros_like(self.Z)

        gl_h, gl_l = puffer.shape
        assert gl_l < self.length and gl_h < self.height, "Glider is bigger than grid"
        mid_start_l = self.length // 2 - gl_l // 2
        mid_end_l = mid_start_l + gl_l

        mid_start_h = self.height // 2 - gl_h // 2
        mid_end_h = mid_start_h + gl_h

        self.Z[mid_start_h:mid_end_h, mid_start_l:mid_end_l] = puffer

class GameOfLife_Shaker(GameOfLife):
    def __init__(self, length: int = 512, height: int = 256):
        super().__init__(length=length, height=height)
        self.title = "shaker"
        puffer = np.array(
[    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
]
        )

        self.Z = np.zeros_like(self.Z)

        gl_h, gl_l = puffer.shape
        assert gl_l < self.length and gl_h < self.height, "Glider is bigger than grid"
        mid_start_l = self.length // 2 - gl_l // 2
        mid_end_l = mid_start_l + gl_l

        mid_start_h = self.height // 2 - gl_h // 2
        mid_end_h = mid_start_h + gl_h

        self.Z[mid_start_h:mid_end_h, mid_start_l:mid_end_l] = puffer
