#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MaxNumVar 4000
#define PreassignVar 18
//Ԥ�ȷ���ı�Ԫ��Ŀ
#define CONFLICT 0
#define SATISFIABLE 1
#define UNSATISFIABLE 0
#define OTHERS 2
#define SINGLE -1
//UNKNOWN��ʾ�ñ�Ԫδ֪��NONE��ʾ�ñ�Ԫ�����ڣ�FALSE��ʾ�ñ�ԪΪ�٣�TRUE��ʾ�ñ�ԪΪ��
#define FALSE -1
#define TRUE 1
#define UNKNOWN 0
#define NONE 2
//�ýṹ�ο�˫���ּ��ӷ������ɿ������ű��
typedef struct varWatch {
    struct varList *pos;       //�����ڽӱ��������ڽӱ��������ڽӱ�
    struct varList *neg;
} Var_watch;

typedef struct varList {
    struct clause *p;     //ָ��һ���Ӿ�
    struct varList *next; //ָ����һ�����������ֵ��Ӿ�
} VarList;

typedef struct clause {
    struct clauseLiteral *p;//ָ��һ���Ӿ��е���һ������
    struct clause *nextClause;//������ڽӱ��������壬Ҫ�Ӷ�ȡCNF�ļ�ʱ����Ч��
} Clause;

typedef struct clauseLiteral {
    int data;//���ֵ�ֵ
    struct clauseLiteral *next;        //ָ���Ӿ��е���һ������
} ClauseLiteral;

typedef struct satAnswer {
    int branchLevel[MaxNumVar + 1];   //��ֵʱ�ľ������߶�
    int value[MaxNumVar + 1];          //TRUE or FALSE or UNKNOWN or NONE
    int searched[MaxNumVar + 1];       //�ѱ������������
    int singleClause[MaxNumVar + 1];  //����Ƿ���ڸñ����ĵ��Ӿ�
} SatAnswer;

//��غ����Ķ��� 
int InitSat(Clause **S, SatAnswer **answer, Var_watch *var_watch, int *branchDecision);//��ʼ��������� 

int Sat();//SATģ��������� 

int LoadCnf(Clause **S, SatAnswer *answer, Var_watch var_watch[], FILE *fp);//����cnf�ļ� 

int GetNum(FILE *fp);//��ȡ���ݣ������ 

int DPLL(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int op, int firstBranch);//DPLL�㷨 

int PrintAnswer(SatAnswer *answer, int result, char filename[100], int duration);//����� 

int PutClause(Clause *ctemp, int var, Var_watch var_watch[]);

int Deduce(int blevel, SatAnswer *answer, Var_watch var_watch[], int branchDecision[], VarList *root);

int SingleClauseDeduce(int blevel, SatAnswer *answer, Var_watch var_watch[], int branchDecision[], VarList **vp);

int NextBranch(int branchDecision[], SatAnswer *answer); 

int Analyse_conflict(int *blevel, int var, SatAnswer *answer);//�����Ӿ�֮��ì�� 

int Sudoku();//����ģ��������� 

int NewSudoku(Clause **S, SatAnswer **answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]);//������ 

int GenerateSudoku(Clause **S, SatAnswer **answer, Var_watch var_watch[], int branchDecision[]);//���������� 

int DigHole(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]);//�ڶ� 

int SolveSudoku(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int sudokuTable[9][9]);//������� 

int dig_watch(int sudokuTable[9][9]);//����������� 

//���ȫ�ֱ����Ķ��� 
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
        printf("\t\t\t      ���˵�                      \n");
        printf("\t\t************************************\n");
        printf("\t\t\t1.  ����                   2.  SAT\n");
        printf("\t\t\t0.  �˳�                          \n");
        printf("\t\t************************************\n");
        printf("\t\t\t��ѡ�����ѡ��[0--2]:             \n");
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
                printf("\t\t\t���ٴ�ѡ�����ѡ��[0--2]:\n");
                scanf("%d", &op);
        }
    }
    return 0;
}

int InitSat(Clause **S, SatAnswer **answer, Var_watch var_watch[], int branchDecision[]) {
    Clause *cfront = *S, *crear;                                   //�Ӿ��ǰ��ָ�룬�������
    ClauseLiteral *lfront, *lrear;                                 //���ֵ�ǰ��ָ��
    //�Ӿ��ʼ��
    while (cfront) {
        crear = cfront->nextClause;                                //�����п������Ϊһ���������Ͻǳ���
        lfront = cfront->p;
        while (lfront) {
            lrear = lfront->next;
            free(lfront);
            lfront = lrear;                                        //��һ����Ӿ��е����֣���ձ���е�һ��
        }
        free(cfront);
        cfront = crear;                                            //������֮������free��ǰָ�롣����
    }
    *S = NULL;
    //���ʼ��
    numVar = 0;                                                   //��Ԫ��ĿΪ0
    knownVar = 0;                                                 //��֪��Ԫ��ĿΪ0
    numBranch = 0;                                                //��֧��ĿΪ0
    *answer = (SatAnswer *) malloc(sizeof(SatAnswer));            //��answer�ṹ����ռ�
    for (int i = 1; i <= MaxNumVar; ++i) {
        //���ʼ��
        (*answer)->value[i] = NONE;                               //��ʾ��Ԫ������
        (*answer)->branchLevel[i] = 0;                           //��ֵʱ�������߶�Ϊ0
        (*answer)->searched[i] = 0;                              //�ѱ��������������0
        (*answer)->singleClause[i] = 0;                          //�ñ����ĵ��Ӿ���ĿΪ0
        //�����ڽӱ��ʼ��
        var_watch[i].pos = (VarList *) malloc(sizeof(VarList));  //Ϊ�������ڽӱ�ָ�����ռ�
        var_watch[i].pos->next = NULL;                           //�������ڽӱ�ָ��ָ���
        var_watch[i].neg = (VarList *) malloc(sizeof(VarList));  //Ϊ�������ڽӱ�ָ�����ռ�
        var_watch[i].neg->next = NULL;                           //�������ڽӱ�ָ��ָ���
    }
    //��֧���߼�������ʼ��
    for (int j = 1; j <= 2 * MaxNumVar; ++j)
        branchDecision[j] = 0;                                  //��֧���߼�������ʼ��Ϊ0
}

int Sat() {
    Clause *S = NULL,*p=NULL;                           //�Ӿ�ָ��
    ClauseLiteral *q=NULL;
    SatAnswer *answer;                          //SAT���ָ��
    Var_watch var_watch[MaxNumVar + 1];         //����Ԫ��Ŀ��һ���ռ�
    FILE *fp;								    //�ļ�ָ��
    char filename[100];						    //�ļ���
    int branchDecision[2 * MaxNumVar + 1];      //����������֧ Ӧ���Ǳ�Ԫ��Ŀ�Ķ�����һ��һ����Ԫ�����ֿ��ܣ�
    int op = 1,result;
    clock_t start, finish;                      //���õ�time.hͷ�ļ�
    int duration;                               //���ڱ�ʾ��ʱ
    while (op) {
        printf("\n\n");
        printf("\t\t\t\t      SAT                               \n");
        printf("\t\t\t******************************************\n");
        printf("\t\t\t1.  ��SAT                         0.  �˳�\n");
        printf("\t\t\t******************************************\n");
        printf("\t\t\t���������ѡ��[0--2]:\n            "         );
        scanf("%d", &op);
        system("cls");
        switch (op) {
            case 1://case1��ζ��Ҫ��ȡcnf�ļ�������Ҫ��ʼ�����������һ��
                InitSat(&S, &answer, var_watch, branchDecision);
                printf("�������ļ�·��:\n");
                scanf("%s", filename);
                fp = fopen(filename, "r");
                if (fp == NULL) {
                    printf("δ�ܴ��ļ�!\n ");
                    break;
                } else LoadCnf(&S, answer, var_watch, fp);      //����LoadCnf ��������ȡ�ļ�
                printf("��cnf�ļ�������Ϊ��\n");
                p=S;
                for(;p!=NULL;p=p->nextClause){
                	for(q=p->p;q!=NULL;q=q->next)
                		printf("%d ",q->data);
                	printf("\n");
                }
                start = clock();                              //����time.h����¼��ʼʱ��
                result = DPLL(answer, var_watch, branchDecision, 1, 1);
                finish = clock();                             //����time.h����¼����ʱ��
                duration = (finish - start);//�õ��������ĺ�ʱ
                if (result == SATISFIABLE)
				{
					printf("���SAT����ʱ����%d ms\n", duration);
					PrintAnswer(answer, 1, filename, duration);
				}
                else
				{
					printf("�޽�!\n");
				}
                break;
            case 0:
                return 0;
            default:
                printf("\t\t\t���ٴ�����ѡ��[0~2]:\n");
                scanf("%d", &op);
        }
    }
}

int LoadCnf(Clause **S, SatAnswer *answer, Var_watch var_watch[], FILE *fp) {//����CNF�ļ�
    char c;
    Clause *ctemp, *cp = NULL;          //����ָ�룬����CNF�ļ��е��Ӿ���õ�
    ClauseLiteral *lp, *ltemp;         //����ָ�룬����CNF�ļ��е����ֻ��õ�
    int var;                            //������ֵ
    int numClauseVar;                   //ÿ���Ӿ�ı�Ԫ��
    fscanf(fp, "%c", &c);
    while (c == 'c') {                  //ע�Ͷ�
        while (c != '\n' && c != '\r')  //
            fscanf(fp, "%c", &c);       //
        fscanf(fp, "%c", &c);
        if (c == '\n')                  //�����ȡ�����з�����ζ��Ҫ���ظ�һ��
            fscanf(fp, "%c", &c);
    }
    fscanf(fp, " cnf ");                //��ʼ�����ļ�����Ϣ��
    numVar = GetNum(fp);              //��ȡ������������ֵ��numVar
    GetNum(fp);                        //��ȡ�Ӿ���
    var = GetNum(fp);                  //�ٴε��ã���ȡ��һ�е�һ������ֵ����ֵ����var
    while (1) {
        numClauseVar = 0;
        ctemp = (Clause *) malloc(sizeof(Clause));      //Ϊctempָ�����ռ�
        lp = ctemp->p;                                  //lpΪ����ָ�룬����������һ�±��ʲô��
        while (var) {
            ++numClauseVar;                             //���������ܣ�ͳ��ÿ���Ӿ�ı�Ԫ��Ŀ
            if (answer->value[abs(var)] == NONE)
                answer->value[abs(var)] = UNKNOWN;
            ltemp = (ClauseLiteral *) malloc(sizeof(ClauseLiteral));
            ltemp->data = var;                          //���ֵ�ֵ��ĳ�var����ֵ
            ltemp->next = NULL;                         //
            if (numClauseVar == 1) {                    //�����Ӿ����׸�����
                ctemp->p = lp = ltemp;
            } else {                                    //�����Ӿ��з��׸�����
                lp->next = ltemp;
                lp = lp->next;
            }
            if (var > 0)
                ++firstBranch[var];                    //��ʼ��֧���߼�������
            else
                ++firstBranch[numVar - var];
            PutClause(ctemp, var, var_watch);             //������������Ӿ��ַ
            var = GetNum(fp);
        }
        if (numClauseVar == 1) {                        //���뵥�Ӿ䣬����Ӿ�������㣬�������
            answer->value[abs(lp->data)] = lp->data / abs(lp->data);
            ++knownVar;                                //��֪��Ԫ��Ŀ��1
        } else if (*S == NULL) {
            *S = cp = ctemp;
            cp->nextClause = NULL;
        } else {                                        //���������ʽ�����е�ͷ
            cp->nextClause = ctemp;
            cp = cp->nextClause;
            cp->nextClause = NULL;
        }
        var = GetNum(fp);                             //�������ļ�β����ִ��һ��z'z ���ļ�����ʱ�������ļ�������־
        if (feof(fp))
            break;
    }
}

int GetNum(FILE *fp) {//�������������ȡcnf�ļ�ʱ�õ����������Ӿ��������������
    char c;
    int sign = 1, num = 0;        //num �����õ����ֵ�ֵ,sign������������������߸�
    fscanf(fp, "%c", &c);
    if (c == '-') {
        sign = -1;                //sign��Ϊ-1����ʾΪ������
        fscanf(fp, "%c", &c);
    } else if (c == '0') {        //��ʾ�����Ӿ����
        fscanf(fp, "%c", &c);
        if (c == '\r')            //��ʾ����
            fscanf(fp, "%c", &c);
        return num;               //
    } else if (feof(fp))          //����ǽ������
        return 0;
    while (c != ' ' && c != '\n' && c != '\r') {
        num = num * 10 + c - '0'; //�õ����ֵ�ֵ
        fscanf(fp, "%c", &c);
    }
    if (c == '\r')
        fscanf(fp, "%c", &c);
    return sign * num;                      //�����õ����֣�����������ֵ��
}

int DPLL(SatAnswer *answer, Var_watch var_watch[], int branchDecision[], int op, int firstBranch) {
    int status, var, blevel = 0;            //��ʼ�ж���Ϊ0
    VarList *vp;                            //�ڽӱ�ָ��
    while (1) {
        if (numBranch++ == 0) {             //��һ�η�֧����
            if (op == 1)                    //1:SAT��⡢2:���������ļ������
                var = NextBranch(branchDecision, answer);
            else
                var = firstBranch;
        } else
            var = NextBranch(branchDecision, answer);        //��һ��֧����
        ++blevel;                                            //�ж�����1
        answer->value[abs(var)] = var / abs(var);            //����һ��֧
        answer->branchLevel[abs(var)] = blevel;
        ++answer->searched[abs(var)];                        //�ѱ������������1
        ++knownVar;                                          //��֪��Ԫ��Ŀ��1
        while (1) {
            if (var > 0)
                vp = var_watch[var].neg->next;               //varΪTRUE��������varΪFALSE���Ӿ�
            else
                vp = var_watch[-var].pos->next;              //varΪFALSE��������varΪTRUE���Ӿ�
            status = Deduce(blevel, answer, var_watch, branchDecision, vp);//���Ӿ䴫���������Ӿ��״̬
            if (status == SATISFIABLE)                       //��������
                return SATISFIABLE;
            else if (status == CONFLICT) {
                var = Analyse_conflict(&blevel, var, answer);//var > 0��ì�ܣ���ʼ����
                if (blevel == 0)
                    return UNSATISFIABLE;
                else {                                       //������һ��֧��������
                    answer->value[var] = -answer->value[var];//��ֵ���з�ת������ɸ����������
                    ++answer->searched[var];                 //�����������Ŀ��1
                    if (answer->value[var] < 0)
                        var = -var;
                }
            } else if (status == OTHERS) break;              //��֪�������㣬������һ��
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
    printf("���Ѿ�������\n");
    printf("SAT����һ������:\n");
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
    if (var > 0)                                     //�ж�var�Ƿ�����㣬�Ӷ����ɵ���Ӧ�ı����
        wp = var_watch[var].pos;
    else
        wp = var_watch[-var].neg;
    while (wp->next)
        wp = wp->next;                               //ѭ�����ҵ�VarList��β������var��ӵ�β��
    wp->next = (VarList *) malloc(sizeof(VarList));//����ռ�
    wp = wp->next;                                   //
    wp->p = ctemp;                                   //���Ӿ����ĩβ
    wp->next = NULL;
}

int Deduce(int blevel, SatAnswer *answer, Var_watch var_watch[], int branchDecision[], VarList *root) {
    int top = 0, status;                               //status��ʾ״̬
    VarList *stack[MaxNumVar], *vp = root;             //ջ��ջ����󳤶�Ϊ����Ԫ��Ŀ
    stack[top++] = vp;
    while (top) {
        vp = stack[top - 1];                           //����ջ��Ԫ��
        status = SINGLE;
        while (status == SINGLE && vp) {               //����������
            status = SingleClauseDeduce(blevel, answer, var_watch, branchDecision, &vp);
            stack[top++] = vp;                         //������ջ
        }
        --top;                                         //��ָ����ջ
        if (status == CONFLICT)
            return CONFLICT;
        if (top) {                                     //����������
            vp = stack[--top];                         //���ڵ��ջ
            if (vp->next)
                stack[top++] = vp->next;               //�Һ�����ջ
        }
    }
    if (knownVar < numVar)
        return OTHERS;
    else return SATISFIABLE;
}

int SingleClauseDeduce(int blevel, SatAnswer *answer, Var_watch var_watch[], int branchDecision[], VarList **vp) {
    Clause *cp;                                      //�Ӿ�ָ��
    ClauseLiteral *lp;                               //����ָ��
    int unknownNum, firstUnknown, satisfiable;       //
	//��ʼ��
    unknownNum = 0;
    firstUnknown = 0;
    satisfiable = 0;
    cp = (*vp)->p;                                   //��cpָ��vp��ָ����Ӿ�
    lp = cp->p;                                      //����ָ��ָ��cp��ָ�������
    if (lp == NULL)
        return OTHERS;
    while (lp) {
        if (lp->data > 0)
            ++branchDecision[lp->data];                         //��֧���߼�������
        else
            ++branchDecision[numVar - lp->data];
        if (answer->value[abs(lp->data)] * lp->data > 0) {      //�Ӿ��д���ֵΪTRUE�����֣��Ӿ����
            satisfiable = 1;
            break;
        }
        if (answer->value[abs(lp->data)] == UNKNOWN) {
            ++unknownNum;                         //�����Ӿ���δ����ֵ�����֣�
            if (firstUnknown == 0)
                firstUnknown = lp->data;          //��¼��һ��δ֪������
        }
        lp = lp->next;
    }
    if (unknownNum == 0 && satisfiable == 0)       //���Ӿ����־���֪���Ҷ�ΪFALSE��Ϊì�ܾ�
        return CONFLICT;
    else if (unknownNum == 1 && satisfiable == 0) {  //���Ӿ���ֵΪTRUE�����֣���ֻ��һ��δ֪���֣�Ϊ���Ӿ�
        answer->singleClause[abs(firstUnknown)] = 1; //��ǣ����Ӿ���ֵ�λ��
        answer->value[abs(firstUnknown)] = firstUnknown / abs(firstUnknown);
        answer->branchLevel[abs(firstUnknown)] = blevel;
        ++knownVar;                                 //�ѱ���ֵ�ı�Ԫ��Ŀ��1,��ȷ����Ԫ��Ŀ��1
        if (firstUnknown > 0)
            *vp = var_watch[firstUnknown].neg->next;  //varΪTRUE�������varΪFALSE���Ӿ�
        else
            *vp = var_watch[-firstUnknown].pos->next; //varΪFALSE�������varΪTRUE���Ӿ�
        return SINGLE;
    } else if (knownVar < numVar) {
        *vp = NULL;
        return OTHERS;                                //�ж��������㣬����OTHERS
    } else return SATISFIABLE;
}

int NextBranch(int branchDecision[], SatAnswer *answer) {//��һ��֧����
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

int Analyse_conflict(int *blevel, int var, SatAnswer *answer) {//���ݺ���
    int fore = abs(var);
    while (*blevel != 0) {
        for (int j = 1; j <= numVar; ++j)
            if (j != fore && answer->branchLevel[j] == *blevel) {      //����var��ֵ�����ĵ��Ӿ�����
                answer->value[j] = UNKNOWN;
                answer->branchLevel[j] = 0;
                answer->searched[j] = 0;
                answer->singleClause[j] = 0;
                --knownVar;
            }
        if (*blevel != 1) {
            if (answer->searched[fore] == 2) {       //var��TRUE��FALSE��֧�������������л���
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
            } else break;           //������һ��֧
        } else if (answer->searched[abs(fore)] == 2)//blevel1ȫ��������
            --(*blevel);
        else break;                 //����blevel1����һ��֧
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
        printf("\t\t\t       ����                               \n");
        printf("\t\t********************************************\n");
        printf("\t\t\t1.  ��������          2.  ��ʼ��Ϸ        \n");
        printf("\t\t\t3.  �鿴��          4.  DPLL���������\n");
        printf("\t\t\t0.  �˳�                                  \n");
        printf("\t\t********************************************\n");
        printf("\t\t\t���ٴ�����ѡ��[0--4]:                     \n");
        scanf("%d", &op);
        system("cls");
        switch (op) {
            case 1:
                start = clock();
                printf("����Ϊ���������������Եȣ�\n");
                NewSudoku(&S, &answer, var_watch, branchDecision, sudokuTable);
                finish = clock();
                duration = (double) (finish - start) / 1000.0;
                printf("��������ʱ���� %.3f s\n", duration);
                break;
            case 2:
            	for(int i=1;i<=81-position_number;i++)
            	{
					dig_watch(sudokuTable);
            		printf("������ϣ�������ֵ�λ�ã����磨1��1����:\n");
            		scanf("%d%d",&x,&y);
            		if(sudokuTable[x-1][y-1]||x<1||x>9||y<1||y>9){
						printf("��λ���Ѿ��������ˣ�������ѡ��\n");
						scanf("%d%d",&x,&y);
					}
           			printf("��λ��������Ϊ:\n");
            		scanf("%d",&num);
            		if(sudokuanswer[x-1][y-1]==num){
            			printf("����������ȷ���������\n");
            			sudokuTable[x-1][y-1]=num;
            		}
            		else 
					{
						printf("�������ֲ���ȷ(����һ�λ���)���Ƿ�鿴�𰸣�����1�鿴�𰸣�����2��������,����3��������\n");
            			scanf("%d",&option);
            			if(option==1){
							printf("�ô���ȷ��Ϊ:%d",sudokuanswer[x-1][y-1]);
							sudokuTable[x-1][y-1]=sudokuanswer[x-1][y-1];
						}
            			else if(option==2){
            				printf("�������µĴ𰸣�\n");
            				scanf("%d",&num);
            				if(sudokuanswer[x-1][y-1]==num){
            					printf("����������ȷ���������\n");
            					sudokuTable[x-1][y-1]=num;
            				}
            				else {
								printf("�������ִ���û�л����ˣ��ô���ȷ��Ϊ%d\n",sudokuanswer[x-1][y-1]);
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
					printf("��ϲ�㣡���������\n");
            		dig_watch(sudokuanswer);
            	}
				else printf("���ٽ�������\n");
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
                printf("�����������ʱ����%.3f s\n", duration);
                break;
            case 0:
                return 0;
            default:
                printf("\t\t\t���ٴ�����ѡ��[0--4]:\n");
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
                    fprintf(fp, "%d %d 0\r\n", -(81 * x + 9 * y + z), -(81 * x + 9 * y + i));//ÿ��λ�ã�����1~9�������һ��
            }
    for (x = 0; x < 9; ++x)
        for (z = 1; z <= 9; ++z)
            for (y = 0; y < 8; ++y) {
                for (i = y + 1; i < 9; ++i)
                    fprintf(fp, "%d %d 0\r\n", -(81 * x + 9 * y + z), -(81 * x + 9 * i + z));//ÿһ�У�����1~9�������һ��
            }
    for (y = 0; y < 9; ++y)
        for (z = 1; z <= 9; ++z)
            for (x = 0; x < 8; ++x) {
                for (i = x + 1; i < 9; ++i)
                    fprintf(fp, "%d %d 0\r\n", -(81 * x + 9 * y + z), -(81 * i + 9 * y + z));//ÿһ�У�����1~9�������һ��
            }
    for (z = 1; z <= 9; ++z)
        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j) {
                for (x = 0; x < 3; ++x)
                    for (y = 0; y < 3; ++y)
                        fprintf(fp, "%d ", 81 * (3 * i + x) + 9 * (3 * j + y) + z);//����1~9��ÿ��3��3���������ٳ���һ��
                fprintf(fp, "0\r\n");
                for (x = 0; x < 3; ++x) {
                    for (y = 0; y < 3; ++y) {
                        for (k = x + 1; k < 3; ++k)
                            for (l = 0; l < 3; ++l)
                                if (l != y)
                                    fprintf(fp, "%d %d 0\r\n", -(81 * (3 * i + x) + 9 * (3 * j + y) + z),
                                            -(81 * (3 * i + k) + 9 * (3 * j + l) + z));//����1~9��ÿ��3��3�������������һ��
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
        for (j = 81; j > 1; --j) {         //������ɳ�ʼ��˳��
            index = rand() % j + 1;
            if (j != index) {
                dig_order[j] = dig_order[j] ^ dig_order[index];
                dig_order[index] = dig_order[index] ^ dig_order[j];
                dig_order[j] = dig_order[j] ^ dig_order[index];
            }
        }
        for (k = 0; k < 11;) {                 //�����������ѡ11�������������1~9
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
    for (x = 0; x < 9; ++x) {                        //�õ����̽��
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
    for (j = 81; j > 1; --j) {         //��������ڶ�˳��
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
        if (z <= 0)          //��λ�ò����ڣ�Ѱ����һ��λ��
            continue;
        knownVar = 9 * (81 - j);         //�Ѿ��ڵ�j����
        numBranch = 0;
        for (i = 1; i <= 9; ++i)           //��ȥ��λ��
            answer->value[81 * x + 9 * y + i] = UNKNOWN;
        if (j < 4) {             //��ȥ����С��4����ض�Ψһ
            ++j;
            sudokuTable[x][y] = 0;
            for (k = 1; k <= 9; ++k)           //��ȥ��λ��
                answer->value[81 * x + 9 * y + k] = UNKNOWN;
            continue;
        }
        for (i = 1; i <= 9; ++i) {            //�����ȥ��λ�ý��Ƿ�Ψһ
            if (i == z)
                continue;
            firstBranch = 81 * x + 9 * y + i;
            answer->searched[81 * x + 9 * y + i] = 1;       //����i����һ��֧
            if (DPLL(answer, var_watch, branchDecision, 2, firstBranch) == SATISFIABLE)     //��ȥ��λ����������
                break;
            knownVar = 9 * (81 - j);             //�Ѿ��ڵ�j����
            numBranch = 0;
            for (k = 1; k <= numVar; ++k) {        //��������
                if (!answer->branchLevel[k])       //���߼�Ϊ0��Ϊ��ʼ��������������
                    continue;
                answer->value[k] = UNKNOWN;
                answer->branchLevel[k] = 0;
                answer->searched[k] = 0;
                answer->singleClause[k] = 0;
            }
        }
        if (i == 10) {       //��ȥ��λ�ý���Ψһ
            ++j;
            sudokuTable[x][y] = 0;
        } else {            //��ȥ��λ�ýⲻΨһ
            if (dig > 81)
                break;
            sudokuTable[x][y] = -sudokuTable[x][y];         //��λ�ò�����ȥ
            for (k = 1; k <= numVar; ++k) {        //��������
                if (!answer->branchLevel[k])       //���߼�Ϊ0��Ϊ��ʼ��������������
                    continue;
                answer->value[k] = UNKNOWN;
                answer->branchLevel[k] = 0;
                answer->searched[k] = 0;
                answer->singleClause[k] = 0;
            }
            for (k = 1; k <= 9; ++k)            //����ԭ������
                if (k == z)
                    answer->value[81 * x + 9 * y + k] = TRUE;
                else
                    answer->value[81 * x + 9 * y + k] = FALSE;
        }
    }
    fp = fopen("sudoku_rule.txt", "a+");
    printf("������������\"sudokuTable.txt\"\n");
    printf("��%d����֪����:\n", 81 - j + 1);
    position_number=81-j+1;
    for (x = 0; x < 9; ++x) {                    //�õ���������
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
    for (x = 0; x < 9; ++x) {                        //�õ����̽��
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
    for (x = 0; x < 9; ++x) {                    //��ӡ������
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

