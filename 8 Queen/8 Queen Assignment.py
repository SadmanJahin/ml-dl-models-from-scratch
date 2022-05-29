board = [
 (0, 'Q', 0, 0, 0, 0, 0, 'Q'),
 (0, 0, 0, 0, 0, 0, 0, 0),
 (0, 0, 0, 0, 0, 'Q', 0, 0),
 (0, 0, 0, 0, 'Q', 0, 0, 0),
 (0, 0, 'Q', 0, 0, 0, 0, 0),
 ('Q', 0, 0, 0, 0, 0, 0, 0),
 (0, 0, 0, 'Q', 0, 0, 0, 0),
 (0, 0, 0, 0, 0, 0, 'Q', 0)
 ]

COL, ROW = 8, 8
l = []

for i in range(len(board)):
    for j in range(len(board)):
        if board[j][i] == 'Q':
           l.append([(j, i)])

print('The Queens are in these positions: ', l)

right = 0
for queen in l:
    position = queen[0]
    row = position[0]
    col = position[1]
    range_start = col + 1
    range_end = COL
    for i in range(range_start, range_end):
        if board[row][i] == 'Q':
           right = right + 1

print('\nFace to face in the row = ', right)

dia_down = 0
for queen in l:
    position = queen[0]
    row = position[0]
    col = position[1]
    range_start = row + 1
    range_end = COL - col
    j = 1
    for i in range(range_start, range_end):
        if board[row + j][col + j] == 'Q':
           dia_down = dia_down + 1
        j = j + 1

print('Diagonally down (face to face diagonally down )= ', dia_down)

dia_up = 0
for queen in l:
   position = queen[0]
   row = position[0]
   col = position[1]
   range_start = 0
   range_end = row
   j = 1

   for i in reversed(range(range_start, range_end)):
       if i == -1 or col + j == COL:
          break
       if board[i][col + j] == 'Q':
          dia_up = dia_up + 1
       j = j + 1

print('Diagonally up (face to face diagonally up) = ', dia_up)

print("Heuristic 3: ", right+dia_down+dia_up)