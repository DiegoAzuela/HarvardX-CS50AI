"""
Tic Tac Toe Player
"""

import math
import copy
import random
import numpy as np

X = "X"
O = "O"
EMPTY = None

# NUMPY
def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns player who is to play in the board now.
    """
    # Set a counter to see how many moves have happened.
    count = 0
    # If 'count' is even, it's player 'X' turn, otherwise its player 'O' turn.
    for i in range(len(board)):
        for j in range(len(board[i])):
            if (board[i][j]== X) or (board[i][j] == O):
                count += 1
    # In the initial game state, X gets the first move. Subsequently, the player alternates with each additional move.
    if (board == initial_state()) or (count%2==0):
        return X
    # Any return value is acceptable if a terminal board is provided as input (i.e., the game is already over).
    return O


def alt_player(board):
    """
    Returns player is to play on the board next
    """
    if player(board) == X: return O
    else: return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # Possible moves are any cells on the board that do not already have an X or an O in them
    action_set = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if (board[i][j]==EMPTY):
                action_set.add((i,j))
    # Each action should be represented as a tuple (i, j) where i corresponds to the row of the move (0, 1, or 2) and j corresponds to which cell in the row corresponds to the move (also 0, 1, or 2).
    return action_set


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # First check if the action is possible
    if (action not in actions(board)):
        raise Exception("There are no valid actions left for the current game, please try again")
    # Create a deep copy of the board
    temp_board = copy.deepcopy(board)
    # In the new board, place the players action
    temp_board[action[0]][action[1]] = player(board)

    return temp_board


def value_board(board):
    """
    Returns a list of the sum of every column, row and vertical combination
    """
    # Create matrix with assigned values [X=1, O=-1, EMPTY=0]
    value_matrix = [[EMPTY, EMPTY, EMPTY],[EMPTY, EMPTY, EMPTY],[EMPTY, EMPTY, EMPTY]]

    # Assign values to matrix
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == X:
                value_matrix[i][j] = 1
            elif board[i][j] == O:
                value_matrix[i][j] = -1
            else:
                value_matrix[i][j] = 0

    # Sum the values of columns, rows and diagonals
    column_1,column_2,column_3 = value_matrix[0][0]+value_matrix[1][0]+value_matrix[2][0], value_matrix[0][1]+value_matrix[1][1]+value_matrix[2][1], value_matrix[0][2]+value_matrix[1][2]+value_matrix[2][2]
    row_1,row_2,row_3 = value_matrix[0][0]+value_matrix[0][1]+value_matrix[0][2], value_matrix[1][0]+value_matrix[1][1]+value_matrix[1][2], value_matrix[2][0]+value_matrix[2][1]+value_matrix[2][2]
    diagonal_1,diagonal_2= value_matrix[0][0]+value_matrix[1][1]+value_matrix[2][2], value_matrix[2][0]+value_matrix[1][1]+value_matrix[0][2]
    
    # Value for each column/row/diagonal coordinate
    column_1_value,column_2_value,column_3_value = [value_matrix[0][0],value_matrix[1][0],value_matrix[2][0]], [value_matrix[0][1],value_matrix[1][1],value_matrix[2][1]], [value_matrix[0][2],value_matrix[1][2],value_matrix[2][2]]
    row_1_value,row_2_value,row_3_value = [value_matrix[0][0],value_matrix[0][1],value_matrix[0][2]], [value_matrix[1][0],value_matrix[1][1],value_matrix[1][2]], [value_matrix[2][0],value_matrix[2][1],value_matrix[2][2]]
    diagonal_1_value,diagonal_2_value = [value_matrix[0][0],value_matrix[1][1],value_matrix[2][2]], [value_matrix[2][0],value_matrix[1][1],value_matrix[0][2]]
    
    # Coordinate for each option
    column_1_coord,column_2_coord,column_3_coord = [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)]
    row_1_coord,row_2_coord,row_3_coord = [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)]
    diagonal_1_coord, diagonal_2_coord = [(0,0),(1,1),(2,2)],[(2,0),(1,1),(0,2)]


    return [[column_1,column_2,column_3,row_1,row_2,row_3,diagonal_1,diagonal_2],[[column_1_value,column_2_value,column_3_value],[row_1_value,row_2_value,row_3_value],[diagonal_1_value,diagonal_2_value]],[[column_1_coord,column_2_coord,column_3_coord],[row_1_coord,row_2_coord,row_3_coord],[diagonal_1_coord,diagonal_2_coord]]]


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    possible_winners = value_board(board)[0]

    # There is a winner if the sum of values is either 3 or -3
    for possible_winner in possible_winners:
        if possible_winner == 3:
            return X
        if possible_winner == -3:
            return O 
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    board = np.array(board)
    if (EMPTY not in board) or (winner(board)):
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        if winner(board) == X:
            return 1
        if winner(board) == O:
            return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # If there is no more moves, we cannot continue
    if terminal(board):
        print("The game is over, winner is Player ", winner(board))
        return None
    for action in actions(board):
        # Returns the action that would make the player and inmediate winner
        if terminal(result(board,action)):
            return action
        # for action in actions(temp_board):
        # if temp_board[action[0]][action[1]] = alt_player(temp_board)
        else:
            # For player's 'X' turn, we want to look for '1' in the value board
            if player(board) == X:
                wanted_value = 1
            # For player's 'X' turn, we want to look for '-1' in the value board
            elif player(board) == O:
                wanted_value = -1

            # The following routine applies when the game has already started, not the starting move
            try:
                # We look for the wanted value and obtain its index. This will give us the value of the column, row, diagonal we could set up our next action
                index_1 = value_board(board)[0].index(wanted_value)
                # Obtain the coordinates (column/row/diagonal) that has that value
                for option in value_board(board)[1][index_1]:
                    index_2 = value_board(board)[1][index_1].index(option)
                    # Count the amount of Zeros present
                    count = 0
                    for i in range(len(option)):
                        if option[i] == 0:
                            count += 1
                            # Select an action that has 2 empty spaces
                            if count > 1:
                                index_3 = i
                                return value_board(board)[2][index_1][index_2][index_3]
                       
            except: 
                # If its the first move, select randomly
                return random.choice(list(actions(board)))