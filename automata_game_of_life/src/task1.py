import numpy as np
from PIL import Image, ImageDraw


class Automata:
    def __init__(self, width: int = 12, height: int = 100, rule_num: int = 30, reversable: bool = False, central: bool = True):

        self.central = central
        self.rev = reversable

        assert width > 0 and height > 0, "Width and Height should be greater than 0"
        self.width = width
        self.height = height

        self.init_cond = self._initial_condition()

        assert rule_num < 256, "There only 256 rules"
        assert rule_num > 0, "Number should be greater than 0"
        self.rule_num = rule_num

        self.rule = self._generate_rule()

        self.states = [[1,1,1], [1,1,0], [1,0,1], [1,0,0], [0,1,1], [0,1,0], [0,0,1], [0,0,0]]

    def _initial_condition(self):
        if self.central:
            data = np.zeros(self.width, dtype=np.int32).tolist()
            data[len(data)//2] = 1
        else:
            data = np.random.randint(0, 2, self.width).tolist()
        return data

    def _generate_rule(self):
        binar = list(bin(self.rule_num)[2:])
        for _ in range(8 - len(binar)):
            binar.insert(0, '0')
        return [int(c) for c in binar]

    def _generate_row(self, arr):
        row = []
        for i in range(self.width):
            id = self.states.index([arr[(i-1)%self.width], arr[i], arr[(i+1)%self.width]])
            num = self.rule[id]
            row.append(num)
        return row

    def _generate_row_reversable(self, arr, arr_1):
        row = []
        for i in range(self.width):
            id = self.states.index([arr[(i-1)%self.width], arr[i], arr[(i+1)%self.width]])
            num = self.rule[id]
            row.append((num+arr_1[i])%2)
        return row

    def generate_automata(self):

        game_array = [self.init_cond]

        if self.rev:
            game_array.insert(0, self._initial_condition())
            for i in range(1, self.height):
                game_array.append(self._generate_row_reversable(game_array[i], game_array[i-1]))
            game_array = game_array[1:]

        else:
            for i in range(self.height-1):
                game_array.append(self._generate_row(game_array[i]))

        return game_array

    def text_result(self):
        result = self.generate_automata()
        result_str = []
        for i in range(self.height):
            result_str.append(''.join(map(str, result[i])))
            # print(result_str[i])
        return result_str

    def plot_result(self):
        fname = f"img/rule_{self.rule_num}_width_{self.width}_height_{self.height}.png"
        width = self.width
        height = self.height
        img = Image.new("RGB",(width,height),(255,255,255))
        draw = ImageDraw.Draw(img)
        data = self.generate_automata()
        for y in range(height):
            for x in range(width):
                if data[y][x]: draw.point((x,y),(0,0,0))
        print(f"RULE: {self.rule_num}")
        if self.central:
            print(f"Initialization Central")
        else:
            print(f"Initialization Random")

        if self.rev:
            print("Reversable")
        else:
            print("Non-Reversable")
        img.show()
        img.save(fname,"PNG")

    def animate_result(self):
        images = []
        width = self.width
        height = self.height
        data = self.generate_automata()
        img = Image.new("RGB",(width,height),(255,255,255))
        draw = ImageDraw.Draw(img)
        for y in range(height):
            for x in range(width):
                if data[y][x]: draw.point((x,y),(0,0,0))
            images.append(img.copy())
        images[0].save('pillow_imagedraw.gif',
                   save_all = True, append_images = images[1:],
                   optimize = True, duration = 10)



class Random_Automata(Automata):

    def __init__(self):
        self.width=1000
        self.height=1000

        if np.random.randint(0, 2, 1).tolist()[0] == 1:
            self.rev = True
        else:
            self.rev = False

        if np.random.randint(0, 2, 1).tolist()[0] == 1:
            self.central = True
            self.init_cond = self._initial_condition()
        else:
            self.central = False
            self.init_cond = self._initial_condition()

        self.rule_num = np.random.randint(0, 256, 1).tolist()[0]

        self.rule = self._generate_rule()

        self.states = [[1,1,1], [1,1,0], [1,0,1], [1,0,0], [0,1,1], [0,1,0], [0,0,1], [0,0,0]]
