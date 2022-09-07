#使用博弈树、估价函数，确定电脑最佳落子处

#from _typeshed import BytesPath


def printboard(board):
    
    #显示棋盘当前状态
    
    boardshow='''
                ----------------
                |    |    |    |
                |  %s | %s  | %s  |
                |    |    |    |
                ----------------
                |    |    |    |
                |  %s | %s  | %s  |
                |    |    |    |
                ----------------
                |    |    |    |
                |  %s | %s  | %s  | 
                |    |    |    |
                ----------------    
            '''
    dropshow=[]
    for c in board:
        if c==0:
            dropshow.append(' ')
        elif c==1:
            dropshow.append('O')
        else:
            dropshow.append('X')
    print(boardshow%tuple(dropshow[6:9]+dropshow[3:6]+dropshow[0:3]))


def playerselect(board):
    
    #玩家落子，改变棋盘状态即可
    position=int(input('请选择你要落子的位置(1-9):'))
    while position not in range(1,10) or board[position-1]!=0:
        position=int(input('非法落子位置，请重新选择(1-9):'))
    return position-1


def chesscountvalue(chesscount):
    
    #返回对应棋子状态的估价
    num1=chesscount[1] 
    num2=chesscount[2]
    num0=chesscount[0]
    if num1==3:
        return 10
    elif num2==3:
        return -10
    elif num1==2 and num0==1:
        return 5
    elif num1==1 and num0==2:
        return 1
    elif num2==2 and num0==1:
        return -5
    elif num2==1 and num0==2:
        return -1
    return 0


def evaluate(board):
    
    #估价函数，判断当前棋盘的估价值
    
    # 判断每一行、每一列、每条对角线的棋子状态，不同的状态有不同的估价值
    board=[board[6:9],board[3:6],board[0:3]] 
    chess=[0,0,0] # 记录棋子状态
    value=0
    for i in range(0,3):
        chess=[0,0,0]
        for j in range(0,3):
            chess[board[i][j]]+=1 # 判断行
        
        tempvalue=chesscountvalue(chess)
        if abs(tempvalue)==10:
            return tempvalue
        else:
            value+=tempvalue
        
        chess=[0,0,0]
        for j in range(0,3):
            chess[board[j][i]]+=1 # 判断列
        
        tempvalue=chesscountvalue(chess)
        if abs(tempvalue)==10:
            return tempvalue
        else:
            value+=tempvalue
        
    chess=[0,0,0]
    for p in range(0,3):
        chess[board[p][p]]+=1 # 判断正对角线

    tempvalue=chesscountvalue(chess)
    if abs(tempvalue)==10:
        return tempvalue
    else:
        value+=tempvalue
    
    chess=[0,0,0]
    for p in range(0,3):
        chess[board[p][2-p]]+=1 # 判断负对角线

    tempvalue=chesscountvalue(chess)
    if abs(tempvalue)==10:
        return tempvalue
    else:
        value+=tempvalue
        
    return value
 
def searchTree(board,whosturn,alpha,beta):
    
    #深度优先搜索，在落子一个位置后，直到棋盘满，这个过程产生的估价值
    
    #maxvalue=-100 # 用来存储对当前落子方来说最大的价值
    #minvalue=100 # 用来存储对当前落子方来说最小的价值
    
    value=evaluate(board) # 计算当前棋盘状态的价值
    isfull=True # 棋盘是否已经下满
    if abs(value)>=10: # 若当前棋盘状态已经达到最大值或最小值，返回即可
        return value
    
    # 遍历每个棋盘位置
    #for i in range(0,9):
    #    if board[i]!=0: # 不等于0表示已经有棋子落在了该处，则继续判断下一个位置
    #        continue
    #    isfull=False # 执行到这一步说明棋盘没有满，将isFull赋值为False
    #    board[i]=whosturn # 将当前落子方的棋子下在当前遍历的位置

        # 递归调用，继续向下进行搜索
    #    tempvalue=searchTree(board,((whosturn-1)^1)+1)
    #    maxvalue=max(maxvalue,tempvalue) # 判断下在当前遍历的位置后产生的价值是否大于maxValue
    #    minvalue=min(minvalue,tempvalue) # 判断下在当前遍历的位置后产生的价值是否小于minValue
    #    board[i]=0 # 将被落子的位置清空，继续循环判断下在其他位置后棋盘的价值
    #if isfull: # 若isFull为True，表示棋盘已满，返回进入该函数时棋盘的价值
    #    return value
    #if whosturn==1: # 若当前落子方为电脑，返回最大价值
    #    return maxvalue
    #if whosturn==2: # 若当前落子方为玩家，返回最小价值
    #    return minvalue
    for i in range(0,9):
        if board[i]!=0: # 不等于0表示已经有棋子落在了该处，则继续判断下一个位置
            continue
        isfull=False # 执行到这一步说明棋盘没有满，将isFull赋值为False
        board[i]=whosturn # 将当前落子方的棋子下在当前遍历的位置
        tempvalue=searchTree(board,((whosturn-1)^1)+1,alpha,beta)
        board[i]=0 
        if whosturn==1:
            if tempvalue>alpha:
                alpha=tempvalue
            if alpha>=beta:
                return beta
        if whosturn==2:
            if tempvalue<beta:
                beta=tempvalue
            if beta<=alpha:
                return alpha
    if isfull: # 若isFull为True，表示棋盘已满，返回进入该函数时棋盘的价值
        return value
    if whosturn==1:
        return alpha
    if whosturn==2:
        return beta



def computerselect(board):
    
    #遍历棋盘，计算在可以落子的位置落子后的估价函数值，选择值最高的位置落子
    
    mapgrid=[]
    for i in range(0,9):
        if board[i]!=0:
            continue
        board[i]=1
        mapgrid.append([searchTree(board,2,-100,100),i])
        board[i]=0
    mapgrid=sorted(mapgrid,key=lambda x: x[0],reverse=True)
    return mapgrid[0][1]


def winner(board):
    
    #判断刚刚落子的一方是否获胜（电脑胜出还是玩家胜出）
    
    value=evaluate(board)
    if value==10:
        return 1 # 电脑获胜
    elif value==-10:
        return 2 # 玩家获胜
    else:
        for i in range(0,9):
            if board[i]==0:
                return 0
        return 3


def game():
    
    #主循环，电脑玩家依次落子，电脑每次落子前选择最优位置落子，玩家根据输入落子
    
    continuegame=True
    while continuegame:
        whofirst=(input('谁执先手？(输入1电脑先手，玩家先手请按其他键):'))
        computersturn=1
        playersturn=2
        whosturn=computersturn if whofirst=='1' else playersturn

        # 0表示可以落子，1表示电脑已落子，2表示玩家已落子
        board=[0,0,0,0,0,0,0,0,0]

        # 游戏开始
        gaming=True
        printboard(board)
        while gaming:
            if whosturn==playersturn:
                playerdrop=playerselect(board)
                board[playerdrop]=2
                whosturn=computersturn
            elif whosturn==computersturn:
                computerdrop=computerselect(board)
                board[computerdrop]=1
                whosturn=playersturn
            printboard(board)
            state=winner(board)
            if state==1:
                print('电脑获胜！')
            elif state==2:
                print('你获胜！')
            elif state==3:
                print('平局！')
            if state!=0:
                continueGame=(input('继续游戏请输入1，退出请按其他键：'))
                if continueGame=='1':
                    continuegame=True
                else:
                    continuegame=False
                break

game()