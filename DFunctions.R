trank <- function(x) 
{
  x_unique <- unique(x)
  x_ranks <- rank(x_unique, ties.method = "max")
  tx <- x_ranks[match(x,x_unique)] - 1
  
  tx <- tx / length(unique(tx))
  tx <- tx / max(tx)
  
  return(tx)
}


quantile_normalize_bart <- function(X) 
{
  apply(X = X, MARGIN = 2, trank)
}


preprocess_df <- function(X) 
{
  stopifnot(is.data.frame(X))
  X <- model.matrix(~.-1, data = X)
  group <- attr(X, "assign") - 1
  return(list(X = X, group = group))
  
}



GetSigma <- function(X,Y) 
{ 
  require("glmnet")
  stopifnot(is.matrix(X) | is.data.frame(X))
  
  if(is.data.frame(X)) 
  {
    X <- model.matrix(~.-1, data = X)
  }
  fit <- cv.glmnet(x = X, y = Y)
  fitted <- predict(fit, X)
  sigma_hat <- sqrt(mean((fitted - Y)^2))
  return(sigma_hat)
}


unnormalize_bart <- function(z, a, b) {
  y <- (b - a) * (z + 0.5) + a
  return(y)
}




writeBartMpiFiles=function(numnodes,x,y,xp=NULL,wt=NULL,xroot="x",yroot="y",xproot="xp",wtroot="wt",verbose=FALSE) 
{
  
  filelist = NULL 
  p=ncol(x)
  n=nrow(x)
  
  pv=rep(0,numnodes-1) 
  pv[1:(numnodes-1)] = floor(n/(numnodes)) 
  pv[numnodes]=n-sum(pv)
  
  if (verbose) cat("Generating ",numnodes," train datasets of sizes ",pv,"\n")
  for(i in 1:(numnodes)) 
  {
    if(i==1) {
      ii=1:pv[1]
    } else {
      ii=(sum(pv[1:(i-1)])+1):sum(pv[1:i])
    }
    
    xx=x[ii,]
    yy=y[ii]

    tempfile = paste(c(yroot,xroot),i-1,'.csv',sep="") 


    write.table(yy,tempfile[1],row.names=FALSE,col.names=FALSE,sep=",")
    write.table(xx,tempfile[2],row.names=FALSE,col.names=FALSE,sep=",") 
    if(!is.null(wt))
    {
      wtd=wt[ii]
      wtfile = paste(wtroot,i-1,'.csv',sep="") 
      write.table(wtd,wtfile,row.names=FALSE,col.names=FALSE,sep=",")
      tempfile=c(tempfile,wtfile)
      
    }
    filelist = c(filelist,tempfile)
  }
  
  if(!is.null(xp)) 
  {
      pv=rep(0,numnodes)
      pv[1:(numnodes-1)] = floor(nrow(xp)/numnodes)
      pv[numnodes]=nrow(xp)-sum(pv)
      
      if (verbose) cat("Generating ",numnodes," predictive X datasets of sizes ",pv,"\n")
  
      
      for(i in 1:numnodes) #c indices of slaves
      {
        if(i==1) 
        {
          ii=1:pv[1]
        } else 
        {
          ii=(sum(pv[1:(i-1)])+1):sum(pv[1:i])
        }
        
        xx=xp[ii,]
        tempfile = paste(xproot,i-1,'.csv',sep="") 

        write.table(xx,tempfile,row.names=FALSE,col.names=FALSE,sep=",")            
        filelist = c(filelist,tempfile)
      }    
  }
  return(filelist)
}







PSBARTB<-function(
  numnodes,
  CPath,
  Prefix,
  X,
  Y,
  X_TEST=NULL,
  WT=NULL,
  group=NULL,
  alpha=1,
  beta=2,
  gamma=0.95,
  k=2,
  sigma_hat=NULL,
  shape=1,
  width=0.1,
  num_tree=50,
  alpha_scale=NULL,
  alpha_shape_1=0.5,
  alpha_shape_2=1,
  tau_rate=10,
  num_tree_prob=NULL,
  temperature=1.0,
  num_burn = 100, 
  num_thin = 1, 
  num_save = 100, 
  num_print = 500,
  update_sigma_mu = TRUE, 
  update_s = TRUE,
  update_alpha = TRUE,
  update_beta = FALSE,
  update_gamma = FALSE,
  update_tau = TRUE,
  update_tau_mean = FALSE,
  verbose=TRUE,
  binary=FALSE,
  binaryOffset=NULL,
  rmfiles=TRUE,
  pause=TRUE)
{ 
  
  if (verbose) cat("***** Running MPIBart\n")
  

  
  if(!is.null(X_TEST) && nrow(X_TEST)<numnodes) 
  {
    X_TEST=NULL
    cat(paste('@@@@@@@@@@@@@ prediction x has less than two observations, set to NULL\n'))
  }
  

 
  if (binary)
  {
    
    if(is.factor(Y)) {
      if(length(levels(Y)) != 2) stop("Y is a factor with number of levels != 2")
      Y = as.numeric(Y)-1
    } 
    else if((length(unique(Y)) == 2) & (max(Y) == 1) & (min(Y) == 0)) 
    {
        cat('NOTE: assumming numeric response is binary\n')
    }
    else
    {
      stop("Y is not suitable for binary anlaysis")
      
    }
    if(length(binaryOffset)==0) binaryOffset=qnorm(mean(Y))
  }
  else
  {
    if((!is.vector(Y)) || (typeof(Y)!="double")) stop("argument Y must be a double vector")
    if(length(unique(Y)) == 2 )  stop("argument Y must be a Non binary vector")
  }  
  
 
  

  

  if(is.null(alpha_scale))    alpha_scale    <-  ncol(X)
  if(is.null(num_tree_prob))  num_tree_prob  <-  2.0 / num_tree
  if (binary)
  {
    sigma_mu       <-  3 / (k * sqrt(num_tree))    
  }
  else
  {
    sigma_mu       <-  0.5 / (k * sqrt(num_tree))
  }  
  

  
  if(is.null(group)) 
  {
    group          <-  1:ncol(X) - 1
  } else 
  {
    group          <-  group - 1
  }
  
  if (!binary)
  {  
    YMin           <- min(Y)
    YMax           <- max(Y)
    YS             <- (Y - YMin) / (YMax - YMin) - 0.5    
    
    
    
    if(is.null(sigma_hat))      sigma_hat      <-  GetSigma(X,YS)
    sigma          <-  sigma_hat
    
    
    
    if (verbose) cat('sighat: ',sigma_hat,'\n') 
    
    
  
  
  }
  else
  {
    YS<-Y
    YMin           <- 0
    YMax           <- 1   
    sigma_hat=1
    sigma =1
  }  
 
  MINMAX<-as.data.frame(c(1:2))
  MINMAX[1,1]<-YMax
  MINMAX[2,1]<-YMin
  write.table(MINMAX,paste(Prefix,"MM.csv",sep=""),row.names=FALSE,col.names=FALSE,sep=",") 
  
  
  n <- nrow(X)
  idx_train <- 1:n
  X_TRANS <- rbind(X, X_TEST)
   
  if(is.data.frame(X_TRANS)) {
    cat("Preprocessing data frame\n")
    preproc_df <- preprocess_df(X_TRANS)
    X_TRANS <- preproc_df$X
  }
  
  if (is.null(WT))
  {
    WT=rep(1,n)
  }
  
  
  X_TRANS <- quantile_normalize_bart(X_TRANS)
  X <- X_TRANS[idx_train,,drop=FALSE]
  X_TEST <- X_TRANS[-idx_train,,drop=FALSE]
  
  

   
  
  xroot=paste(Prefix,"x",sep="")
  yroot=paste(Prefix,"y",sep="")
  xproot=paste(Prefix,"xp",sep="")
  wtroot=paste(Prefix,"wt",sep="")
  
  if (verbose)
  {
    cat('xroot,yroot,xproot,wtroot:\n')
    print(xroot);print(yroot);print(xproot);print(wtroot)
  }
  tempfiles = writeBartMpiFiles(numnodes,X,YS,X_TEST,WT,xroot,yroot,xproot,wtroot,verbose=verbose)                              
  write(group,ncol=1,file=paste(Prefix,"group.txt",sep=""))                              
  
  if (binary)
  {
    update_sigma_mu<-0
  }
  else
  {
    update_sigma_mu<-ifelse(update_sigma_mu,1,0)
  }
 
  if(!binary) {
    binaryOffset = -1000.0
  } 
  
  
  binary         <-ifelse(binary,1,0)    
  update_s       <-ifelse(update_s,1,0)
  update_alpha   <-ifelse(update_alpha,1,0)
  update_beta    <-ifelse(update_beta,1,0)
  update_gamma   <-ifelse(update_gamma,1,0)
  update_tau     <-ifelse(update_tau,1,0)
  update_tau_mean<-ifelse(update_tau_mean,1,0)
  verbose        <-ifelse(verbose,1,0)
  
  if (verbose) cat('*****running mcmc code in parallel\n')
  cmd = paste('mpiexec -np ',numnodes,' ',CPath,'bdsbart',sep='')
  cmd = paste(cmd,Prefix,alpha,beta,gamma,binary,sigma_hat)
  cmd = paste(cmd,shape,width,num_tree,alpha_scale,alpha_shape_1)
  cmd = paste(cmd,alpha_shape_2,tau_rate,binaryOffset,temperature,sigma_mu,num_burn,num_thin)
  cmd = paste(cmd,num_save,num_print ,update_sigma_mu,update_s,update_alpha,update_beta,update_gamma,update_tau,update_tau_mean,verbose)
  if (verbose) 
  {
    cat('cmd:\n')
    cat(cmd)
    cat('\n')
  }
  #  if (pause) readline("hit <enter> to continue")
  
  
  ptime = system.time({system(cmd)})   
  
  #ptime      
  
  fit<-list()
  
  fname= paste(Prefix,'_R_sigma.csv',sep="") 
  temp<-read.csv(fname,sep=',',header=FALSE)
  fit$sigma= temp[,1]  
  fit$sigma_Mu= temp[,2]  
  
  for(i in 0:(numnodes-1)) 
  { 
    if (i==0)
    {
      ytrain<-read.csv(paste(Pre,'_R_ytrain',i,'.csv',sep="") ,sep=',',header=FALSE)   
      ytest <-read.csv(paste(Pre,'_R_ytest',i,'.csv',sep="") ,sep=',',header=FALSE)   
    }
    else 
    {
      ytrain<-rbind(ytrain,read.csv(paste(Pre,'_R_ytrain',i,'.csv',sep="") ,sep=',',header=FALSE))
      ytest<-rbind(ytest,read.csv(paste(Pre,'_R_ytest',i,'.csv',sep="") ,sep=',',header=FALSE))   
    }
  }
  
  fit$ytrain <-as.vector(as.numeric(unlist(ytrain)))
  fit$ytest <-as.vector(as.numeric(unlist(ytest)))

  fit$binaryOffset <- binaryOffset
  
  
  class(fit) <- "softbart"
  
  return(fit)                              
}



DSBART<-function(numnodes,mpath,Pre,X,y,Xp,WT,num_save,num_burn,ntree)
{
  t1=proc.time()
  parfit=PSBARTB(numnodes,mpath,Pre,X,y,Xp,WT=WT,num_save=num_save,num_burn=num_burn,num_tree=ntree,binary=FALSE)
  t2=proc.time()
  t=t2-t1
  cat(paste0('SBart Cost ',t[3][[1]],' Seconds'))
  cat("\n")
  
  
  
  
  for(i in 0:(numnodes-1)) 
  { 
    if (i==0)
    {
      ytrain<-read.csv(paste(Pre,'_R_ytrain',i,'.csv',sep="") ,sep=',',header=FALSE)   
      ytest <-read.csv(paste(Pre,'_R_ytest',i,'.csv',sep="") ,sep=',',header=FALSE)   
    }
    else 
    {
      ytrain<-rbind(ytrain,read.csv(paste(Pre,'_R_ytrain',i,'.csv',sep="") ,sep=',',header=FALSE))
      ytest<-rbind(ytest,read.csv(paste(Pre,'_R_ytest',i,'.csv',sep="") ,sep=',',header=FALSE))   
    }
  }
  
  
  fname= paste(Pre,'_R_sigma.csv',sep="") 
  temp<-read.csv(fname,sep=',',header=FALSE)
  sigma= temp[,1]  
  
  
  
  system(paste("rm ",Pre,"*",sep=""))
  fit<-list()
  
  fit$y_hat_train_mean <- as.vector(as.numeric(unlist(ytrain)))
  fit$y_hat_test_mean <- as.vector(as.numeric(unlist(ytest)))
  fit$sigma <- mean(sigma)
  
  
  class(fit) <- "DSBART"
  fit
} 