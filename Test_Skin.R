library("glmnet")
library(ggplot2)

mpath="/home/RH/psbart/"
Pre="B_"
setwd(mpath)
source("DFunctions.R")



skin<-read.csv("skin.csv" ,sep=',',header=FALSE)  
set.seed(99)
ratio=0.8
length(skin[,1])
sub<-sample(1:nrow(skin),round(nrow(skin)*ratio))
length(sub)
data_train<-skin[sub,]
data_test<-skin[-sub,]
dim(data_train)
dim(data_test) 











n=nrow(data_train)
np=nrow(data_test) 

num_save=1000
num_burn=1000
ntree=50
numnodes=10


X=data_train[,1:3] 
Xp=data_test[,1:3] 



y<-data_train[,4]
yp<-data_test[,4]





t1=proc.time()
parfit=PSBARTB(numnodes,mpath,Pre,X,y,Xp,num_save=num_save,num_burn=num_burn,num_tree=ntree,binary=TRUE)
t2=proc.time()
t=t2-t1
print(paste0('SBart',t[3][[1]],'S'))




R<-cbind(parfit$ytest,yp)

write.table(R,"RSoft.csv",row.names=FALSE,col.names=TRUE,sep=",")

