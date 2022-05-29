import random;
#position=[6,3,7,0,5,3,1,4]
#print("found for",temp+1, position[i]+1)
list=[]
row=[]
score=[]
count=0
def moveRightDiagonalUp(position):
    global count
    for i in range(0,len(position),1):
        temp=position[i]
        for j in range(i+1, len(position), 1):
            temp-=1
            if temp==position[j]:
               count+=1

def moveRightDiagonalDown(position):
    global count
    for i in range(0,len(position),1):
        temp=position[i]
        for j in range(i+1, len(position), 1):
            temp+=1
            if temp==position[j]:
               count+=1

def moveRight(position):
    global count
    for i in range(0,len(position),1):
        temp=position[i]
        for j in range(i+1, len(position), 1):
            if temp==position[j]:
               count+=1

def moveLeft(position):
    global count
    for i in range(0,len(position),1):
        temp=position[i]
        for j in range(i-1,-1, -1):
            if temp==position[j]:
               count+=1

def moveLeftDiagonalUp(position):
    global count
    for i in range(0,len(position),1):
        temp=position[i]
        for j in range(i-1,-1, -1):
            temp-=1
            if temp==position[j]:
               count+=1

def moveLeftDiagonalDown(position):
    global count
    for i in range(0,len(position),1):
        temp=position[i]
        for j in range(i-1,-1, -1):
            temp+=1
            if temp==position[j]:
               count+=1


def countScore(position):
        moveLeftDiagonalUp(position)
        moveLeftDiagonalDown(position)
        moveLeft(position)
        moveRightDiagonalUp(position)
        moveRightDiagonalDown(position)
        moveRight(position)

        attackerPair=count/2
        if(attackerPair.is_integer()):
           score.append(attackerPair)
        else:
           return

def createList():
    num=15
    queens=8
    for j in range(num):
        row=[]
        for i in range(0,queens,1):
            rand_x=random.randint(0, 7)
            row.append(rand_x)
        list.append(row)
    #print(list)


def EightQueensProblemGeneticAlgorithm():
    global count
    createList()
    for loop in range(10000):
        score.clear()
        for position in list:
            count = 0
            countScore(position)
        Sort()

        list.pop(len(list)-1)

        SelectionAndCrossover()


        print("Final Score: ",score[0])




def Sort():
    for i in range(len(score)):
        for j in range(len(score)):
              if score[i]<score[j]:
                  temp=score[i]
                  temp2=list[i]
                  score[i]=score[j]
                  list[i]=list[j]
                  score[j]=temp
                  list[j]=temp2;

iteraton=0
crossoverList=[]
pivot=4
def SelectionAndCrossover():
   childList1=list[iteraton].copy()
   childList2 =list[iteraton+1].copy()
   temp=childList1[pivot:8]
   childList1[pivot:8]=childList2[pivot:8]
   childList2[pivot:8]=temp
   SwapMutation(childList1,childList2)


def SwapMutation(childList1,childList2):
    rand_x=random.randint(0, 7)
    rand_y = random.randint(0, 7)
    temp1=childList1[rand_x]
    childList1[rand_x]=childList1[rand_y]
    childList1[rand_y]=temp1;

    temp2=childList2[rand_x]
    childList2[rand_x] = childList2[rand_y]
    childList2[rand_y] = temp2

    list.append(childList1)



EightQueensProblemGeneticAlgorithm()