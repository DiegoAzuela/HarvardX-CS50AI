"""
SOURCE
    https://cs50.harvard.edu/ai/2020/projects/0/tictactoe/

SOLVED BY
    Diego Arnoldo Azuela Rosas

UNDERSTANDING
    There are two main files in this project: runner.py and tictactoe.py. tictactoe.py contains all of the logic for playing the game, and for making optimal moves. runner.py has been implemented for you, and contains all of the code to run the graphical interface for the game. Once you’ve completed all the required functions in tictactoe.py, you should be able to run python runner.py to play against your AI!
    Let’s open up tictactoe.py to get an understanding for what’s provided. First, we define three variables: X, O, and EMPTY, to represent possible moves of the board.
    The function initial_state returns the starting state of the board. For this problem, we’ve chosen to represent the board as a list of three lists (representing the three rows of the board), where each internal list contains three values that are either X, O, or EMPTY. What follows are functions that we’ve left up to you to implement!

SPECIFICATION
    Complete the implementations of player, actions, result, winner, terminal, utility, and minimax.
        The player function should take a board state as input, and return which player’s turn it is (either X or O).
            In the initial game state, X gets the first move. Subsequently, the player alternates with each additional move.
            Any return value is acceptable if a terminal board is provided as input (i.e., the game is already over).
        The actions function should return a set of all of the possible actions that can be taken on a given board.
            Each action should be represented as a tuple (i, j) where i corresponds to the row of the move (0, 1, or 2) and j corresponds to which cell in the row corresponds to the move (also 0, 1, or 2).
            Possible moves are any cells on the board that do not already have an X or an O in them.
            Any return value is acceptable if a terminal board is provided as input.
        The result function takes a board and an action as input, and should return a new board state, without modifying the original board.
            If action is not a valid action for the board, your program should raise an exception.
            The returned board state should be the board that would result from taking the original input board, and letting the player whose turn it is make their move at the cell indicated by the input action.
            Importantly, the original board should be left unmodified: since Minimax will ultimately require considering many different board states during its computation. This means that simply updating a cell in board itself is not a correct implementation of the result function. You’ll likely want to make a deep copy of the board first before making any changes.
        The winner function should accept a board as input, and return the winner of the board if there is one.
            If the X player has won the game, your function should return X. If the O player has won the game, your function should return O.
            One can win the game with three of their moves in a row horizontally, vertically, or diagonally.
            You may assume that there will be at most one winner (that is, no board will ever have both players with three-in-a-row, since that would be an invalid board state).
            If there is no winner of the game (either because the game is in progress, or because it ended in a tie), the function should return None.
        The terminal function should accept a board as input, and return a boolean value indicating whether the game is over.
            If the game is over, either because someone has won the game or because all cells have been filled without anyone winning, the function should return True.
            Otherwise, the function should return False if the game is still in progress.
        The utility function should accept a terminal board as input and output the utility of the board.
            If X has won the game, the utility is 1. If O has won the game, the utility is -1. If the game has ended in a tie, the utility is 0.
            You may assume utility will only be called on a board if terminal(board) is True.
        The minimax function should take a board as input, and return the optimal move for the player to move on that board.
            The move returned should be the optimal action (i, j) that is one of the allowable actions on the board. If multiple moves are equally optimal, any of those moves is acceptable.
            If the board is a terminal board, the minimax function should return None.
    For all functions that accept a board as input, you may assume that it is a valid board (namely, that it is a list that contains three rows, each with three values of either X, O, or EMPTY). You should not modify the function declarations (the order or number of arguments to each function) provided.
    Once all functions are implemented correctly, you should be able to run python runner.py and play against your AI. And, since Tic-Tac-Toe is a tie given optimal play by both sides, you should never be able to beat the AI (though if you don’t play optimally as well, it may beat you!)

HINTS
    If you’d like to test your functions in a different Python file, you can import them with lines like from tictactoe import initial_state.
    You’re welcome to add additional helper functions to tictactoe.py, provided that their names do not collide with function or variable names already in the module.
    Alpha-beta pruning is optional, but may make your AI run more efficiently!
"""


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
    Returns player who has the next turn on a board.
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
        raise Exception("This is not a valid action for the current game, please try again")
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
    diagonal_1_coord,diagonal_2_coord = [(0,0),(1,1),(2,2)],[(2,0),(1,1),(0,2)]


    return [[column_1,column_2,column_3,row_1,row_2,row_3],[[column_1_value,column_2_value,column_3_value],[row_1_value,row_2_value,row_3_value],[diagonal_1_value,diagonal_2_value]],[[column_1_coord,column_2_coord,column_3_coord],[row_1_coord,row_2_coord,row_3_coord],[diagonal_1_coord,diagonal_2_coord]]]


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
        print("The game is over, the winner was Player ", winner(board))
        return None
    for action in actions(board):
        # Returns the action that would make the player and inmediate winner
        if terminal(result(board,action)):
            return action
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

trial = [[EMPTY, EMPTY, EMPTY],
        [EMPTY, O, EMPTY],
        [X, EMPTY, EMPTY]]

print(player(trial))