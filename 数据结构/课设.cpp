#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MaxNumVar 4000
#define PreassignVar 18
//预先分配的变元数目
#define CONFLICT 0
#define SATISFIABLE 1
#define UNSATISFIABLE 0
#define OTHERS 2
#define SINGLE -1
//UNKNOWN表示该变元未知，NONE表示该变元不存在，FALSE表示该变元为假，TRUE表示该变元为真
#define FALSE -1
#define TRUE 1
#define UNKNOWN 0
#define NONE 2
//该结构参考双文字监视方法，可看作两张表格
typedef struct varWatch {
    struct varList *pos;       //文字邻接表，正文字邻接表，负文字邻接表，
    struct varList *neg;
} Var_watch;

typedef struct varList {
    struct clause *p;     //指向一个子句
    struct varList *next; //指向下一个包含该文字的子句
} VarList;

typedef struct clause {
    struct clauseLiteral *p;//指向一个子句中的下一个文字
    struct clause *nextClause;//这个在邻接表中无意义，要从读取CNF文件时看出效果
} Clause;

typedef struct clauseLiteral {
    int data;//文字的值
    struct clauseLiteral *next;        //指向子句中的下一个文字
} ClauseLiteral;

typedef struct satAnswer {
    int branchLevel[MaxNumVar + 1];   //赋值时的决策树高度
    int value[MaxNumVar + 1];          //TRUE or FALSE or UNKNOWN or NONE
    int searched[MaxNumVar + 1];       //已被搜索的情况数
    int singleClause[MaxNumVar + 1];  //标记是否存在该变量的单子句
} SatAnswer;

//相关函数的定义 
int InitSat(Clause **S, SatAnswer **answer, Var_watch *var_watch, int *branchDecision);//初始化相关数据 

int Sat();//SAT模块操作界面 

int LoadCnf(Clause **S, SatAnswer *answer, Var_watch var_watch[], FILE *fp);//加载cnf文件 

int GetNum(FILE *fp);//读取数据（快读） 

int DPLL(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int op, int firstBranch);//DPLL算法 

int PrintAnswer(SatAnswer *answer, int result, char filename[100], int duration);//输出答案 

int PutClause(Clause *ctemp, int var, Var_watch var_watch[]);

int Deduce(int blevel, SatAnswer *answer, Var_watch var_watch[], int branchDecision[], VarList *root);

int SingleClauseDeduce(int blevel, SatAnswer *answer, Var_watch var_watch[], int branchDecision[], VarList **vp);

int NextBranch(int branchDecision[], SatAnswer *answer); 

int Analyse_conflict(int *blevel, int var, SatAnswer *answer);//分析子句之间矛盾 

int Sudoku();//数独模块操作界面 

int NewSudoku(Clause **S, SatAnswer **answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]);//新数独 

int GenerateSudoku(Clause **S, SatAnswer **answer, Var_watch var_watch[], int branchDecision[]);//产生新数独 

int DigHole(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]);//挖洞 

int SolveSudoku(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]);//求解数独 

int dig_watch(int sudokuTable[9][9]);//输出数独棋盘 

//相关全局变量的定义 
int numVar;

int knownVar;

int numBranch;

int firstBranch[MaxNumVar];

int position_number;

int sudokuanswer[9][9];

int main() {
    int op = 1;
    while (op) {
        printf("\n\n");
        printf("\t\t\t      主菜单                      \n");
        printf("\t\t************************************\n");
        printf("\t\t\t1.  数独                   2.  SAT\n");
        printf("\t\t\t0.  退出                          \n");
        printf("\t\t************************************\n");
        printf("\t\t\t请选择你的选择[0--2]:             \n");
        scanf("%d", &op);
        system("cls");
        switch (op) {
            case 1:
                Sudoku();
                break;
            case 2:
                Sat();
                break;
            case 0:
                exit(0);
            default:
                printf("\t\t\t请再次选择你的选择[0--2]:\n");
                scanf("%d", &op);
        }
    }
    return 0;
}

int InitSat(Clause **S, SatAnswer **answer, Var_watch var_watch[], int branchDecision[]) {
    Clause *cfront = *S, *crear;                                   //子句的前后指针，用于清空
    ClauseLiteral *lfront, *lrear;                                 //文字的前后指针
    //子句初始化
    while (cfront) {
        crear = cfront->nextClause;                                //这两行可以理解为一个表格从左上角出发
        lfront = cfront->p;
        while (lfront) {
            lrear = lfront->next;
            free(lfront);
            lfront = lrear;                                        //逐一清空子句中的文字，清空表格中的一行
        }
        free(cfront);
        cfront = crear;                                            //清除完毕之后，由于free了前指针。。。
    }
    *S = NULL;
    //解初始化
    numVar = 0;                                                   //变元数目为0
    knownVar = 0;                                                 //已知变元数目为0
    numBranch = 0;                                                //分支数目为0
    *answer = (SatAnswer *) malloc(sizeof(SatAnswer));            //给answer结构分配空间
    for (int i = 1; i <= MaxNumVar; ++i) {
        //解初始化
        (*answer)->value[i] = NONE;                               //表示变元不存在
        (*answer)->branchLevel[i] = 0;                           //赋值时决策树高度为0
        (*answer)->searched[i] = 0;                              //已被搜索的情况数是0
        (*answer)->singleClause[i] = 0;                          //该变量的单子句数目为0
        //文字邻接表初始化
        var_watch[i].pos = (VarList *) malloc(sizeof(VarList));  //为正文字邻接表指针分配空间
        var_watch[i].pos->next = NULL;                           //正文字邻接表指针指向空
        var_watch[i].neg = (VarList *) malloc(sizeof(VarList));  //为负文字邻接表指针分配空间
        var_watch[i].neg->next = NULL;                           //负文字邻接表指针指向空
    }
    //分支决策计数器初始化
    for (int j = 1; j <= 2 * MaxNumVar; ++j)
        branchDecision[j] = 0;                                  //分支决策计数器初始化为0
}

int Sat() {
    Clause *S = NULL,*p=NULL;                           //子句指针
    ClauseLiteral *q=NULL;
    SatAnswer *answer;                          //SAT解的指针
    Var_watch var_watch[MaxNumVar + 1];         //最大变元数目加一个空间
    FILE *fp;								    //文件指针
    char filename[100];						    //文件名
    int branchDecision[2 * MaxNumVar + 1];      //决策树最大分支 应该是变元数目的二倍加一（一个变元有两种可能）
    int op = 1,result;
    clock_t start, finish;                      //调用的time.h头文件
    int duration;                               //用于表示耗时
    while (op) {
        printf("\n\n");
        printf("\t\t\t\t      SAT                               \n");
        printf("\t\t\t******************************************\n");
        printf("\t\t\t1.  新SAT                         0.  退出\n");
        printf("\t\t\t******************************************\n");
        printf("\t\t\t请输入你的选择[0--2]:\n            "         );
        scanf("%d", &op);
        system("cls");
        switch (op) {
            case 1://case1意味着要读取cnf文件，首先要初始化，因此有下一行
                InitSat(&S, &answer, var_watch, branchDecision);
                printf("请输入文件路径:\n");
                scanf("%s", filename);
                fp = fopen(filename, "r");
                if (fp == NULL) {
                    printf("未能打开文件!\n ");
                    break;
                } else LoadCnf(&S, answer, var_watch, fp);      //调用LoadCnf 函数，读取文件
                printf("该cnf文件中数据为：\n");
                p=S;
                for(;p!=NULL;p=p->nextClause){
                	for(q=p->p;q!=NULL;q=q->next)
                		printf("%d ",q->data);
                	printf("\n");
                }
                start = clock();                              //调用time.h，记录开始时间
                result = DPLL(answer, var_watch, branchDecision, 1, 1);
                finish = clock();                             //调用time.h，记录结束时间
                duration = (finish - start);//得到解决问题的耗时
                if (result == SATISFIABLE)
				{
					printf("解决SAT问题时间是%d ms\n", duration);
					PrintAnswer(answer, 1, filename, duration);
				}
                else
				{
					printf("无解!\n");
				}
                break;
            case 0:
                return 0;
            default:
                printf("\t\t\t请再次输入选择[0~2]:\n");
                scanf("%d", &op);
        }
    }
}

int LoadCnf(Clause **S, SatAnswer *answer, Var_watch var_watch[], FILE *fp) {//加载CNF文件
    char c;
    Clause *ctemp, *cp = NULL;          //两个指针，加载CNF文件中的子句会用到
    ClauseLiteral *lp, *ltemp;         //两个指针，加载CNF文件中的文字会用到
    int var;                            //变量的值
    int numClauseVar;                   //每个子句的变元数
    fscanf(fp, "%c", &c);
    while (c == 'c') {                  //注释段
        while (c != '\n' && c != '\r')  //
            fscanf(fp, "%c", &c);       //
        fscanf(fp, "%c", &c);
        if (c == '\n')                  //如果读取到换行符，意味着要再重复一次
            fscanf(fp, "%c", &c);
    }
    fscanf(fp, " cnf ");                //开始读到文件的信息段
    numVar = GetNum(fp);              //读取变量数，并赋值给numVar
    GetNum(fp);                        //读取子句数
    var = GetNum(fp);                  //再次调用，读取下一行第一个变量值，把值赋给var
    while (1) {
        numClauseVar = 0;
        ctemp = (Clause *) malloc(sizeof(Clause));      //为ctemp指针分配空间
        lp = ctemp->p;                                  //lp为文字指针，，可以想象一下表格长什么样
        while (var) {
            ++numClauseVar;                             //计数器功能，统计每个子句的变元数目
            if (answer->value[abs(var)] == NONE)
                answer->value[abs(var)] = UNKNOWN;
            ltemp = (ClauseLiteral *) malloc(sizeof(ClauseLiteral));
            ltemp->data = var;                          //文字的值域改成var，赋值
            ltemp->next = NULL;                         //
            if (numClauseVar == 1) {                    //储存子句中首个变量
                ctemp->p = lp = ltemp;
            } else {                                    //储存子句中非首个变量
                lp->next = ltemp;
                lp = lp->next;
            }
            if (var > 0)
                ++firstBranch[var];                    //初始分支决策计数增加
            else
                ++firstBranch[numVar - var];
            PutClause(ctemp, var, var_watch);             //储存各变量的子句地址
            var = GetNum(fp);
        }
        if (numClauseVar == 1) {                        //输入单子句，则该子句必须满足，无需存入
            answer->value[abs(lp->data)] = lp->data / abs(lp->data);
            ++knownVar;                                //已知变元数目加1
        } else if (*S == NULL) {
            *S = cp = ctemp;
            cp->nextClause = NULL;
        } else {                                        //想想表格的形式，是列的头
            cp->nextClause = ctemp;
            cp = cp->nextClause;
            cp->nextClause = NULL;
        }
        var = GetNum(fp);                             //若到达文件尾，再执行一次z'z 读文件操作时，设置文件结束标志
        if (feof(fp))
            break;
    }
}

int GetNum(FILE *fp) {//这个函数用来读取cnf文件时得到变量数、子句数、后面的内容
    char c;
    int sign = 1, num = 0;        //num 用来得到文字的值,sign用来标记文字是正或者负
    fscanf(fp, "%c", &c);
    if (c == '-') {
        sign = -1;                //sign变为-1，表示为负文字
        fscanf(fp, "%c", &c);
    } else if (c == '0') {        //表示该条子句结束
        fscanf(fp, "%c", &c);
        if (c == '\r')            //表示换行
            fscanf(fp, "%c", &c);
        return num;               //
    } else if (feof(fp))          //如果是结束标记
        return 0;
    while (c != ' ' && c != '\n' && c != '\r') {
        num = num * 10 + c - '0'; //得到文字的值
        fscanf(fp, "%c", &c);
    }
    if (c == '\r')
        fscanf(fp, "%c", &c);
    return sign * num;                      //用来得到文字（包括正负和值）
}

int DPLL(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int op, int firstBranch) {
    int status, var, blevel = 0;            //初始判定级为0
    VarList *vp;                            //邻接表指针
    while (1) {
        if (numBranch++ == 0) {             //第一次分支决策
            if (op == 1)                    //1:SAT求解、2:生成数独的检验求解
                var = NextBranch(branchDecision, answer);
            else
                var = firstBranch;
        } else
            var = NextBranch(branchDecision, answer);        //下一分支决策
        ++blevel;                                            //判定级加1
        answer->value[abs(var)] = var / abs(var);            //进入一分支
        answer->branchLevel[abs(var)] = blevel;
        ++answer->searched[abs(var)];                        //已被搜索情况数加1
        ++knownVar;                                          //已知变元数目加1
        while (1) {
            if (var > 0)
                vp = var_watch[var].neg->next;               //var为TRUE，则搜索var为FALSE的子句
            else
                vp = var_watch[-var].pos->next;              //var为FALSE，则搜索var为TRUE的子句
            status = Deduce(blevel, answer, var_watch, branchDecision, vp);//单子句传播，返回子句的状态
            if (status == SATISFIABLE)                       //满足的情况
                return SATISFIABLE;
            else if (status == CONFLICT) {
                var = Analyse_conflict(&blevel, var, answer);//var > 0，矛盾，开始回溯
                if (blevel == 0)
                    return UNSATISFIABLE;
                else {                                       //进入另一分支，不满足
                    answer->value[var] = -answer->value[var];//则值进行反转，正变成负，负变成正
                    ++answer->searched[var];                 //被搜索情况数目加1
                    if (answer->value[var] < 0)
                        var = -var;
                }
            } else if (status == OTHERS) break;              //已知条件不足，进入下一层
        }
    }
}

int PrintAnswer(SatAnswer *answer,int result, char filename[100], int duration) {
    FILE *fp;
	int p = 0;
	while (filename[p] != 0) p++;
	while (filename[p] != '.') p--;
	p++;
	filename[p] = 'r';
	p++;
	filename[p] = 'e';
	p++;
	filename[p] = 's';
	p++;
	filename[p] = 0;
	fp = fopen(filename, "w");
	if(result== 1)
		fprintf(fp, "s 1\r\n");
	else
		fprintf(fp, "s 0\r\n");
	fprintf(fp, "v ");
	for(int i = 1; i < MaxNumVar; i++)
	{
		if(answer->value[i] == TRUE)
			fprintf(fp, "%d ", i);
		else
			fprintf(fp, "-%d ", i);
	}
	fprintf(fp, "\r\n");
	fprintf(fp, "t %d\r\n", duration);
    fclose(fp);
    printf("答案已经被保存\n");
    printf("SAT其中一个解是:\n");
    for (int i = 1; i <= MaxNumVar; ++i)
        if (answer->value[i] == TRUE)
			{
            	printf("%d ", i);
			}
        else if (answer->value[i] == FALSE)
            	printf("-%d ", i);
    printf("\n");
}

int PutClause(Clause *ctemp, int var, Var_watch var_watch[]) {
    VarList *wp;
    if (var > 0)                                     //判断var是否大于零，从而归纳到相应的表格中
        wp = var_watch[var].pos;
    else
        wp = var_watch[-var].neg;
    while (wp->next)
        wp = wp->next;                               //循环，找到VarList的尾部，将var添加到尾部
    wp->next = (VarList *) malloc(sizeof(VarList));//分配空间
    wp = wp->next;                                   //
    wp->p = ctemp;                                   //将子句放在末尾
    wp->next = NULL;
}

int Deduce(int blevel, SatAnswer *answer, Var_watch var_watch[], int branchDecision[], VarList *root) {
    int top = 0, status;                               //status表示状态
    VarList *stack[MaxNumVar], *vp = root;             //栈，栈的最大长度为最大变元数目
    stack[top++] = vp;
    while (top) {
        vp = stack[top - 1];                           //访问栈顶元素
        status = SINGLE;
        while (status == SINGLE && vp) {               //左子树搜索
            status = SingleClauseDeduce(blevel, answer, var_watch, branchDecision, &vp);
            stack[top++] = vp;                         //左孩子入栈
        }
        --top;                                         //空指针退栈
        if (status == CONFLICT)
            return CONFLICT;
        if (top) {                                     //右子树搜索
            vp = stack[--top];                         //根节点出栈
            if (vp->next)
                stack[top++] = vp->next;               //右孩子入栈
        }
    }
    if (knownVar < numVar)
        return OTHERS;
    else return SATISFIABLE;
}

int SingleClauseDeduce(int blevel, SatAnswer *answer, Var_watch var_watch[], int branchDecision[], VarList **vp) {
    Clause *cp;                                      //子句指针
    ClauseLiteral *lp;                               //文字指针
    int unknownNum, firstUnknown, satisfiable;       //
	//初始化
    unknownNum = 0;
    firstUnknown = 0;
    satisfiable = 0;
    cp = (*vp)->p;                                   //将cp指向vp所指向的子句
    lp = cp->p;                                      //文字指针指向cp所指向的文字
    if (lp == NULL)
        return OTHERS;
    while (lp) {
        if (lp->data > 0)
            ++branchDecision[lp->data];                         //分支决策计数增加
        else
            ++branchDecision[numVar - lp->data];
        if (answer->value[abs(lp->data)] * lp->data > 0) {      //子句中存在值为TRUE的文字，子句成立
            satisfiable = 1;
            break;
        }
        if (answer->value[abs(lp->data)] == UNKNOWN) {
            ++unknownNum;                         //计数子句中未被赋值的文字，
            if (firstUnknown == 0)
                firstUnknown = lp->data;          //记录第一个未知的文字
        }
        lp = lp->next;
    }
    if (unknownNum == 0 && satisfiable == 0)       //该子句文字均已知并且都为FALSE，为矛盾句
        return CONFLICT;
    else if (unknownNum == 1 && satisfiable == 0) {  //该子句无值为TRUE的文字，且只有一个未知文字，为单子句
        answer->singleClause[abs(firstUnknown)] = 1; //标记，单子句出现的位置
        answer->value[abs(firstUnknown)] = firstUnknown / abs(firstUnknown);
        answer->branchLevel[abs(firstUnknown)] = blevel;
        ++knownVar;                                 //已被赋值的变元数目加1,即确定变元数目加1
        if (firstUnknown > 0)
            *vp = var_watch[firstUnknown].neg->next;  //var为TRUE，则检索var为FALSE的子句
        else
            *vp = var_watch[-firstUnknown].pos->next; //var为FALSE，则检索var为TRUE的子句
        return SINGLE;
    } else if (knownVar < numVar) {
        *vp = NULL;
        return OTHERS;                                //判断条件不足，返回OTHERS
    } else return SATISFIABLE;
}

int NextBranch(int branchDecision[], SatAnswer *answer) {//下一分支函数
    int maxVar = numVar, maxCount = 0;
    int *branch;
    ++numBranch;
    branch = numBranch == 1 ? firstBranch : branchDecision;
    for (int i = 1; i <= 2 * numVar; ++i) {
        if (i <= numVar && answer->value[i] != UNKNOWN)
            continue;
        if (i > numVar && answer->value[i - numVar] != UNKNOWN)
            continue;
        if (maxCount <= *(branch + i)) {
            maxVar = i;
            maxCount = *(branch + i);

        }
    }
    return maxVar > numVar ? numVar - maxVar : maxVar;
}

int Analyse_conflict(int *blevel, int var, SatAnswer *answer) {//回溯函数
    int fore = abs(var);
    while (*blevel != 0) {
        for (int j = 1; j <= numVar; ++j)
            if (j != fore && answer->branchLevel[j] == *blevel) {      //将由var赋值产生的单子句重置
                answer->value[j] = UNKNOWN;
                answer->branchLevel[j] = 0;
                answer->searched[j] = 0;
                answer->singleClause[j] = 0;
                --knownVar;
            }
        if (*blevel != 1) {
            if (answer->searched[fore] == 2) {       //var的TRUE和FALSE分支均搜索过，进行回溯
                --(*blevel);
                answer->value[fore] = UNKNOWN;
                answer->branchLevel[fore] = 0;
                answer->searched[fore] = 0;
                --knownVar;
                for (int i = 1; i <= numVar; ++i)
                    if (answer->branchLevel[i] == *blevel && answer->singleClause[i] == 0) {
                        fore = i;
                        break;
                    }
            } else break;           //搜索另一分支
        } else if (answer->searched[abs(fore)] == 2)//blevel1全部搜索完
            --(*blevel);
        else break;                 //搜索blevel1的另一分支
    }
    return fore;
}

int Sudoku() {
    Clause *S = NULL;
    SatAnswer *answer;
    FILE *fp;
    char filename[100];
    Var_watch var_watch[MaxNumVar + 1];
    int branchDecision[2 * MaxNumVar + 1];
    int sudokuTable[9][9];
    int op = 1,x,y,num,option,flag=1;
    clock_t start, finish; 
    double duration;
    srand((unsigned) time(NULL));
    while (op) {
        printf("\n\n");
        printf("\t\t\t       数独                               \n");
        printf("\t\t********************************************\n");
        printf("\t\t\t1.  生成数独          2.  开始游戏        \n");
        printf("\t\t\t3.  查看答案          4.  DPLL求解数独答案\n");
        printf("\t\t\t0.  退出                                  \n");
        printf("\t\t********************************************\n");
        printf("\t\t\t请再次重新选择[0--4]:                     \n");
        scanf("%d", &op);
        system("cls");
        switch (op) {
            case 1:
                start = clock();
                printf("正在为您生成数独，请稍等！\n");
                NewSudoku(&S, &answer, var_watch, branchDecision, sudokuTable);
                finish = clock();
                duration = (double) (finish - start) / 1000.0;
                printf("产生数独时间是 %.3f s\n", duration);
                break;
            case 2:
            	for(int i=1;i<=81-position_number;i++)
            	{
					dig_watch(sudokuTable);
            		printf("请输入希望填数字的位置（例如（1，1））:\n");
            		scanf("%d%d",&x,&y);
            		if(sudokuTable[x-1][y-1]||x<1||x>9||y<1||y>9){
						printf("该位置已经有数字了，请重新选择！\n");
						scanf("%d%d",&x,&y);
					}
           			printf("该位置数字填为:\n");
            		scanf("%d",&num);
            		if(sudokuanswer[x-1][y-1]==num){
            			printf("所填数字正确，请继续！\n");
            			sudokuTable[x-1][y-1]=num;
            		}
            		else 
					{
						printf("所填数字不正确(还有一次机会)，是否查看答案（输入1查看答案，输入2继续作答,输入3放弃作答）\n");
            			scanf("%d",&option);
            			if(option==1){
							printf("该处正确答案为:%d",sudokuanswer[x-1][y-1]);
							sudokuTable[x-1][y-1]=sudokuanswer[x-1][y-1];
						}
            			else if(option==2){
            				printf("请输入新的答案！\n");
            				scanf("%d",&num);
            				if(sudokuanswer[x-1][y-1]==num){
            					printf("所填数字正确，请继续！\n");
            					sudokuTable[x-1][y-1]=num;
            				}
            				else {
								printf("所填数字错误，没有机会了，该处正确答案为%d\n",sudokuanswer[x-1][y-1]);
								sudokuTable[x-1][y-1]=sudokuanswer[x-1][y-1];
							}
						}
            			else if(option==3){
							flag=0;
							break;
						}
            		}
            	}
            	if(flag){
					printf("恭喜你！完成数独！\n");
            		dig_watch(sudokuanswer);
            	}
				else printf("请再接再厉！\n");
            	break; 
            case 3:
            	dig_watch(sudokuanswer);
            	break;
            case 4:
                InitSat(&S, &answer, var_watch, branchDecision);
                fp = fopen("sudoku_rule.txt", "r");
                LoadCnf(&S, answer, var_watch, fp);
                fclose(fp);
                start = clock();
                SolveSudoku(answer, var_watch, branchDecision, sudokuTable);
                finish = clock();
                duration = (double) (finish - start) / 1000.0;
                printf("解决数独问题时间是%.3f s\n", duration);
                break;
            case 0:
                return 0;
            default:
                printf("\t\t\t请再次重新选择[0--4]:\n");
                scanf("%d", &op);
        }
    }
}

int NewSudoku(Clause **S, SatAnswer **answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]) {
    if (GenerateSudoku(S, answer, var_watch, branchDecision) == 0)
        return 0;
    DigHole(*answer, var_watch, branchDecision, sudokuTable);
}

int GenerateSudoku(Clause **S, SatAnswer **answer, Var_watch var_watch[], int branchDecision[]) {
    int x, y, z, i, j, k, l;
    int dig_order[82], index;
   	FILE *fp;
    fp = fopen("sudoku_rule.txt", "w");
    fprintf(fp, "p cnf 729 10287\r\n");
    for (x = 0; x < 9; ++x)
        for (y = 0; y < 9; ++y)
            for (z = 1; z <= 8; ++z) {
                for (i = z + 1; i <= 9; ++i)
                    fprintf(fp, "%d %d 0\r\n", -(81 * x + 9 * y + z), -(81 * x + 9 * y + i));//每个位置，数字1~9至多出现一次
            }
    for (x = 0; x < 9; ++x)
        for (z = 1; z <= 9; ++z)
            for (y = 0; y < 8; ++y) {
                for (i = y + 1; i < 9; ++i)
                    fprintf(fp, "%d %d 0\r\n", -(81 * x + 9 * y + z), -(81 * x + 9 * i + z));//每一行，数字1~9至多出现一次
            }
    for (y = 0; y < 9; ++y)
        for (z = 1; z <= 9; ++z)
            for (x = 0; x < 8; ++x) {
                for (i = x + 1; i < 9; ++i)
                    fprintf(fp, "%d %d 0\r\n", -(81 * x + 9 * y + z), -(81 * i + 9 * y + z));//每一列，数字1~9至多出现一次
            }
    for (z = 1; z <= 9; ++z)
        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j) {
                for (x = 0; x < 3; ++x)
                    for (y = 0; y < 3; ++y)
                        fprintf(fp, "%d ", 81 * (3 * i + x) + 9 * (3 * j + y) + z);//数字1~9在每个3×3数独中至少出现一次
                fprintf(fp, "0\r\n");
                for (x = 0; x < 3; ++x) {
                    for (y = 0; y < 3; ++y) {
                        for (k = x + 1; k < 3; ++k)
                            for (l = 0; l < 3; ++l)
                                if (l != y)
                                    fprintf(fp, "%d %d 0\r\n", -(81 * (3 * i + x) + 9 * (3 * j + y) + z),
                                            -(81 * (3 * i + k) + 9 * (3 * j + l) + z));//数字1~9在每个3×3数独中至多出现一次
                    }
                }
            }
    fclose(fp);
    do {
        fp = fopen("sudoku_rule.txt", "r");
        if (fp == NULL) {
            printf("Opening sudoku_rule.txt\" failed.\n ");
            return 0;
        }
        InitSat(S, answer, var_watch, branchDecision);
        LoadCnf(S, *answer, var_watch, fp);
        fclose(fp);
        for (j = 1; j <= 81; ++j)
            dig_order[j] = j;
        for (j = 81; j > 1; --j) {         //随机生成初始化顺序
            index = rand() % j + 1;
            if (j != index) {
                dig_order[j] = dig_order[j] ^ dig_order[index];
                dig_order[index] = dig_order[index] ^ dig_order[j];
                dig_order[j] = dig_order[j] ^ dig_order[index];
            }
        }
        for (k = 0; k < 11;) {                 //在棋盘中随机选11个格子随机填入1~9
            x = (dig_order[j] - 1) / 9;
            y = (dig_order[j] - 1) % 9;
            z = rand() % 9 + 1;
            for (l = 1; l <= 9; ++l)
                if (l == z)
                    (*answer)->value[81 * x + 9 * y + l] = TRUE;
                else
                    (*answer)->value[81 * x + 9 * y + l] = FALSE;
            ++k;
        }
        knownVar = k;
    } while (DPLL(*answer, var_watch, branchDecision, 2, -(rand() % 729 + 1)) == UNSATISFIABLE);
    return 1;
}

int DigHole(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]) {
    int x, y, z, i, j, k;
    int dig_order[82], index, dig = 1;
    int firstBranch;
    FILE *fp;
    clock_t a, b;
    a = clock();
    for (i = 1; i <= numVar; ++i) {
        answer->branchLevel[i] = 0;
        answer->searched[i] = 0;
        answer->singleClause[i] = 0;
    }
    for (x = 0; x < 9; ++x) {                        //得到终盘结果
        for (y = 0; y < 9; ++y) {
            for (z = 1; z <= 9; ++z)
                if (answer->value[81 * x + 9 * y + z] == TRUE) {
                    sudokuTable[x][y] = z;
                    sudokuanswer[x][y]=z;
                    break;
                }
        }
    }
   	for (j = 1; j <= 81; ++j)
        dig_order[j] = j;
    for (j = 81; j > 1; --j) {         //随机生成挖洞顺序
        index = rand() % j + 1;
        if (j != index) {
            dig_order[j] = dig_order[j] ^ dig_order[index];
            dig_order[index] = dig_order[index] ^ dig_order[j];
            dig_order[j] = dig_order[j] ^ dig_order[index];
        }
    }
    for (j = 1; j <= 81 - PreassignVar && dig <= 81;) {
    	dig_watch(sudokuTable);
        x = (dig_order[dig] - 1) / 9;
        y = (dig_order[dig++] - 1) % 9;
        z = sudokuTable[x][y];
        if (z <= 0)          //该位置不可挖，寻找下一个位置
            continue;
        knownVar = 9 * (81 - j);         //已经挖掉j个洞
        numBranch = 0;
        for (i = 1; i <= 9; ++i)           //挖去该位置
            answer->value[81 * x + 9 * y + i] = UNKNOWN;
        if (j < 4) {             //挖去个数小于4，解必定唯一
            ++j;
            sudokuTable[x][y] = 0;
            for (k = 1; k <= 9; ++k)           //挖去该位置
                answer->value[81 * x + 9 * y + k] = UNKNOWN;
            continue;
        }
        for (i = 1; i <= 9; ++i) {            //检测挖去该位置解是否唯一
            if (i == z)
                continue;
            firstBranch = 81 * x + 9 * y + i;
            answer->searched[81 * x + 9 * y + i] = 1;       //锁定i的另一分支
            if (DPLL(answer, var_watch, branchDecision, 2, firstBranch) == SATISFIABLE)     //挖去该位置有其他解
                break;
            knownVar = 9 * (81 - j);             //已经挖掉j个洞
            numBranch = 0;
            for (k = 1; k <= numVar; ++k) {        //重置终盘
                if (!answer->branchLevel[k])       //决策级为0，为初始化条件，不重置
                    continue;
                answer->value[k] = UNKNOWN;
                answer->branchLevel[k] = 0;
                answer->searched[k] = 0;
                answer->singleClause[k] = 0;
            }
        }
        if (i == 10) {       //挖去该位置解仍唯一
            ++j;
            sudokuTable[x][y] = 0;
        } else {            //挖去该位置解不唯一
            if (dig > 81)
                break;
            sudokuTable[x][y] = -sudokuTable[x][y];         //该位置不可挖去
            for (k = 1; k <= numVar; ++k) {        //重置终盘
                if (!answer->branchLevel[k])       //决策级为0，为初始化条件，不重置
                    continue;
                answer->value[k] = UNKNOWN;
                answer->branchLevel[k] = 0;
                answer->searched[k] = 0;
                answer->singleClause[k] = 0;
            }
            for (k = 1; k <= 9; ++k)            //填入原来的数
                if (k == z)
                    answer->value[81 * x + 9 * y + k] = TRUE;
                else
                    answer->value[81 * x + 9 * y + k] = FALSE;
        }
    }
    fp = fopen("sudoku_rule.txt", "a+");
    printf("数独被保存至\"sudokuTable.txt\"\n");
    printf("有%d个已知数字:\n", 81 - j + 1);
    position_number=81-j+1;
    for (x = 0; x < 9; ++x) {                    //得到生成数独
        for (y = 0; y < 9; ++y) {
            sudokuTable[x][y] = abs(sudokuTable[x][y]);
            if (sudokuTable[x][y] != 0) {
                for (i = 1; i <= 9; ++i) {
                    if (i != sudokuTable[x][y])
                        fprintf(fp, "%d 0\r\n", -(81 * x + 9 * y + i));
                    else
                        fprintf(fp, "%d 0\r\n", 81 * x + 9 * y + i);
                }
            }
            if (y != 0 && y % 3 == 0)
                printf("| ");
            if(sudokuTable[x][y])printf("%d ", sudokuTable[x][y]);
            else printf("_ ");
        }
        printf("\n");
        if (x != 8 && x % 3 == 2)
            printf("---------------------\n");
    }
    fclose(fp);
}

int SolveSudoku(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]) {
    int x, y, z;
    DPLL(answer, var_watch, branchDecision, 1, 1);
    for (x = 0; x < 9; ++x) {                        //得到终盘结果
        for (y = 0; y < 9; ++y) {
            for (z = 1; z <= 9; ++z)
                if (answer->value[81 * x + 9 * y + z] == TRUE) {
                    sudokuTable[x][y] = z;
                    break;
                }
        }
    }
    dig_watch(sudokuTable);
}

int dig_watch(int sudokuTable[9][9]) {
    int x, y;
    for (x = 0; x < 9; ++x) {                    //打印数独答案
        for (y = 0; y < 9; ++y) {
            sudokuTable[x][y] = abs(sudokuTable[x][y]);
            if (y != 0 && y % 3 == 0)
                printf("| ");
            if(sudokuTable[x][y])printf("%d ", sudokuTable[x][y]);
            else printf("_ ");
        }
        printf("\n");
        if (x != 8 && x % 3 == 2)
            printf("---------------------\n");
    }
    printf("\n\n");
}

