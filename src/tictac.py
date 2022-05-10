from pickle import TRUE
from common import *
import random
from joblib import load

""" PRINTS TICTACTOE BOARD
    INPUTS:
        board: vector of current board
    RETURNS:
        NA
"""
def print_board(board):
    print(f'-------------------------')
    for i in range(0, 7, 3):
        print(f'|   {board[i]}   |   {board[i + 1]}   |   {board[i + 2]}   |')
        print(f'-------------------------')

""" UPDATES BOARD
    INPUTS:
        board: current board
        move: player move choice
        player: turn(player or computer)
    RETURNS:
        player: Retry or next move
        win: Did someone win
"""
def update_board(board, move, player):
    if(board[move] == 0):
        board[move] = player
        if(check_win(board)):
            return player, True
        else:
            return (player * -1), False
    
    return player, False
        
""" CHECKS IF A PLAYER WON
    INPUTS:
       board: current board
    RETURNS:
        win: bool for if game is won
"""    
def check_win(board):
    if(board[0] == board[3] == board[6] != 0 or board[1] == board[4] == board[7] != 0 or board[2] == board[5] == board[8] != 0):
        return True
    if(board[0] == board[1] == board[2] != 0 or board[3] == board[4] == board[5] != 0 or board[6] == board[7] == board[8] != 0):
        return True
    if(board[0] == board[4] == board[8] != 0 or board[2] == board[4] == board[6] != 0):
        return True
    return False

""" DETERMINES COMPUTER MOVE
    INPUTS:
        moves: possible player moves
        board: current board
    RETURNS:
        move: best move indx
"""
def comp_move(moves, board):
    #moves = moves[0]
    while(not all([ v == 0 for v in moves ])) :
        index = numpy.argmax(moves)
        if(board[index] == 0):
            return index
        moves[index] = 0
    index = [i for i in range(len(board)) if board[i] == 0]
    return random.choice(index)

""" CHECKS FOR DRAW
    INPUTS:
        board: current board
    RETURNS:
        draw: bool for board is in draw state
"""           
def check_draw(board):
    return all([ v != 0 for v in board ])

if __name__ == '__main__':
    X,Y = load_tictac_multi()
    play = True
    while(play):
        #Choose Opponnent
        print(f'1) Linear Regressor\n2) K-NN\n3) MLP')
        reg = int(input('Choose Opponent Regressor: '))
        choice = True
        if(reg == 1):
            clf = numpy.loadtxt("../regressor_models/weights.txt")
        elif(reg == 2):
            clf =  load('../regressor_models/knn.joblib')
        elif(reg == 3):
            clf = load('../regressor_models/mlp.joblib')
        else:
            choice = False

        while(choice):
            board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            print_board(board)
            player = 1
            win = False
            draw = False
            while(not win and not draw):
                # Get Move
                if(player == 1):
                    location = int(input('Player 1 Move: '))
                else:
                    x= []
                    x.append(board)
                    if(reg == 1):
                        location = comp_move(numpy.matmul(board, clf), board) 
                    else:
                        location = comp_move(clf.predict(x).flatten(), board)
                # Attempt Move
                player,win = update_board(board, location, player)
                draw = check_draw(board)
                print_board(board)

            if(draw and not win):
                print("Draw!")    
            elif(player == 1):
                print("Player 1 Won!")
            else:
                print("Player 2 Won!")

            choice = input('Play again? (Y,N):' )
            if(choice == 'N'):
                play = False
            choice = False