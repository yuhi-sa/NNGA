#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define In 2
#define Hidden 3
#define Out 1

#define Number 150 //NNの個体数

#define kousa 0.8 //交叉確率
#define change 0.07
  //突然変異確率

typedef struct{
    double u[In][Hidden]; //入力層-隠れ層の重み
    double v[Hidden]; //隠れ層-出力層の重み
    double bias_h[Hidden]; //隠れ層のバイアス
    double bias_o; //出力層のバイアス
    double F; //適合度
} NN;

typedef struct{
    double input[Number][In];
    double output[Number];
}kyoshisingo;

double x[In]; //入力
double y; //出力
double kansu();
double sigmoid(double x);
double gosa(NN nn1);

NN nn[Number];
kyoshisingo teacher;

void NN_Output(NN nn1);
void make_first_NN();
void make_teacher();
void fitness();
void GA();
int p[2]; //ルーレット選択によって選ばれる個体
int elite;

int main(){
    FILE *fp ;
int count;
int i;
int j;
double kiroku[1000000];
int generation=0; //世代数のカウント

//初期の個体を生成する
make_first_NN();
//教師信号の入出力を決定
make_teacher();
//適合度計算
fitness();


while(1){
    generation++;
    GA();
    fitness();

    double min_E;
    min_E=10000000;

  for(count=0;count<Number;count++){
        // printf("誤差は%f 最小の誤差は%f\n",gosa(nn[count]),min_E);
         if(min_E > gosa(nn[count])){
            min_E=gosa(nn[count]);
            elite=count;
         }        
    }
    kiroku[generation]=min_E;
    printf("%d世代目、誤差は%f\n",generation,min_E);
    
    if(min_E < 0.06) break;
    if(generation==10000) break;
}

//データを保存する
if((fp=fopen("kekka.csv","w"))!=NULL){
    for(int count=0;count<generation;count++){
        //カンマで区切ることでCSVファイルとする
            fprintf(fp,"%f\n,",kiroku[count]);
        }
        //忘れずに閉じる
        fclose(fp);
}

if((fp=fopen("kekka2.csv","w"))!=NULL){
  for(int count=0;count<100;count++){
      for(int i=0;i<In;i++) x[i]=teacher.input[count][i];

        NN_Output(nn[elite]);
        fprintf(fp,"%f,%f,%f\n",teacher.output[count],y,fabs(teacher.output[count]-y));

   }fclose(fp);

}


}


void make_first_NN(){
    for(int count=0;count<Number;count++){
        for(int j=0;j<Hidden;j++){
            for(int i=0;i<In;i++){
                nn[count].u[i][j]=((double)rand()/RAND_MAX)*2-1;
            //    printf("初期個体のu %f\n",nn[count].u[i][j]);
            }
                nn[count].bias_h[j]=((double)rand()/RAND_MAX)*2-1;
                nn[count].v[j]=((double)rand()/RAND_MAX)*2-1;
             //   printf("初期個体のh %f\n",nn[count].bias_h[j]);
        }
                nn[count].bias_o=((double)rand()/RAND_MAX)*2-1;
            //    printf("初期個体のo %f\n",nn[count].bias_o);
    }
}

void make_teacher(){

    for(int count=0;count<Number;count++){
        for(int i=0;i<In;i++){
            teacher.input[count][i]=5*(((double)rand()/RAND_MAX)*2-1);
        }
            teacher.output[count]=kansu(teacher.input[count]);
            //  printf("%3.2d個目の xは %3.2f yは %3.2f 答えは　%3.2f\n",count+1,teacher.input[count][0],teacher.input[count][1],teacher.output[count]);

    }
}

double kansu(double x[In]){
    return((sin(x[0])*sin(x[0])/cos(x[1]))+x[0]*x[0]-5*x[1]+30)/80;
}

double sigmoid(double x){
    return (double)1/(1+exp(-x));
}

void NN_Output(NN nn1){
    int i,j;
    double hidden_node[Hidden];
    y=0;
    for(j=0;j<Hidden;j++){
        hidden_node[j]=0; //初期化

        for(i=0;i<In;i++) {
        hidden_node[j]+=nn1.u[i][j]*x[i];
        hidden_node[j]-=nn1.bias_h[j];
 
        y+=sigmoid(nn1.v[j]*hidden_node[j]);
        }
    }
    y-=nn1.bias_o;
   // printf("yの値は%f\n",y);
}

double gosa(NN nn1){
    double ave_E=0;
    for(int count=0;count<Number;count++){
        for(int i=0;i<In;i++){
            x[i]=teacher.input[count][i];
        }
            NN_Output(nn1);
            ave_E+=(double)fabs(y-teacher.output[count])/Number;
            //printf("平均誤差は %f\n",ave_E);
    }
    return ave_E;
}

void fitness(){
    for(int count=0;count<Number;count++){
        nn[count].F=(double)1/gosa(nn[count]);
        // printf("誤差は%f 適合度は%f\n",gosa(nn[count]),nn[count].F);
    }
}

void GA(){
    int count,i,j;
    double F_sum,F_min,F_ave,F_temp,s;

    // 個体数/2 回行う
    for(count=0;count<ceil((double)Number/2);count++){
        //ルーレット選択
        F_sum=0;
        for(count=0;count<Number;count++){
            F_sum+=nn[count].F; //各個体の適合度を足し合わせている
        // printf("%d 個目の適合度は、%f で累積 %f\n",count,nn[count].F,F_sum);
        }
        for(i=0;i<2;i++){
            F_temp=0;
            j=-1;
            s=((double)rand()/RAND_MAX)*F_sum;

            while(F_temp < s){
                j++;
                F_temp+=nn[count].F;
                //printf("%f\n",F_temp);
            }
            p[i]=j;
        }
        //printf("%d %d %f\n",p[0],p[1],s);

        NN child[2];

        //一様交叉
        if(((double)rand()/RAND_MAX)<kousa){

            for(int j=0;j<Hidden;j++){
                for(int i=0;i<In;i++){
                    if(((double)rand()/RAND_MAX)<0.5){
                        child[0].u[i][j]=nn[p[0]].u[i][j];
                        child[1].u[i][j]=nn[p[1]].u[i][j];                   
                    }else{
                        child[0].u[i][j]=nn[p[1]].u[i][j];
                        child[1].u[i][j]=nn[p[0]].u[i][j];  
                    }
                }
                if(((double)rand()/RAND_MAX)<0.5){
                    child[0].v[j]=nn[p[0]].v[j];
                    child[1].v[j]=nn[p[1]].v[j]; 
                }else{
                    child[0].v[j]=nn[p[1]].v[j];
                    child[1].v[j]=nn[p[0]].v[j];                
                }         
                if(((double)rand()/RAND_MAX)<0.5){
                child[0].bias_h[j]=nn[p[0]].bias_h[j];
                child[1].bias_h[j]=nn[p[1]].bias_h[j]; 
                }else{
                child[0].bias_h[j]=nn[p[1]].bias_h[j];
                child[1].bias_h[j]=nn[p[0]].bias_h[j];    
                }                  
        }
        if(((double)rand()/RAND_MAX)<0.5){
            child[0].bias_o=nn[p[0]].bias_o;
            child[1].bias_o=nn[p[1]].bias_o;    
        }else{
            child[0].bias_o=nn[p[1]].bias_o;
            child[1].bias_o=nn[p[0]].bias_o;           
        }
        }else{
            child[0]=nn[p[0]];
            child[1]=nn[p[1]];
        }
        //親の平均適合度を受け継ぐ
        child[0].F=child[1].F=(double)(nn[p[0]].F+nn[p[1]].F)/2;

        
        //突然変異
    for(int count=0;count < 2;count++){
        for(int j=0;j<Hidden;j++){
            for(int i=0;i<In;i++){
                if(((double)rand()/RAND_MAX) < change) child[count].u[i][j]=((double)rand()/RAND_MAX)*2-1;
            }
            if(((double)rand()/RAND_MAX) < change) child[count].bias_h[j]=((double)rand()/RAND_MAX)*2-1;
            if(((double)rand()/RAND_MAX) < change) child[count].v[j]=((double)rand()/RAND_MAX)*2-1;
        }
        if(((double)rand()/RAND_MAX) < change) child[count].bias_o=((double)rand()/RAND_MAX)*2-1;
    }
        //個体群に子個体を追加


    int rm;
    int rm2=0;
    int rm3=0;
    //rm= nn[1].F;
    rm=100000;

        for(int count=0;count<Number;count++){
            if(rm>nn[count].F){
                rm = nn[count].F;
                rm2 = count;
                //printf("適合度は%f 取り除く値は%f\n",nn[count].F,nn[rm2].F);
            }
        }
            //    printf("適合度%fのを%fのに取り替え\n",nn[rm2],child[0]);
        nn[rm2]=child[0];

    //rm= nn[0].F; 
    rm=100000;  
        for(int count=0;count<Number;count++){
            if(count==rm2){

            }else if(rm>nn[count].F){
                rm = nn[count].F;
                rm3 = count;
                //printf("適合度は%f 取り除く値は%f\n",nn[count].F,nn[rm2].F);
            }
        }
        nn[rm3]=child[1];
            //    printf("適合度%fのを%fのに取り替え\n",nn[rm2],child[1]);
    }

}

