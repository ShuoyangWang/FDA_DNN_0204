#' Functional data via Deep neural network
#'
#' This function trains the deep neural network for functional inputs.
#' 
#' @param Data List of n, each element is a d dimension array.
#' @param d Dimension of data.
#' @param Grid List of d, each gives a vector of selected grid points.
#' @param N Vector of length d, the ith element is the number of grid points in the ith dimension, no more than 4. 
#' @param n sample size.
#' @param p Vector for widths, each layer has the same width.
#' @param L Vector for number of hidden layers.
#' @param s Vector for L1 regularization of factor, from 0 to 1.
#' @param epoch Number of epochs.
#' @param batch Batch size.
#' @param k Number of folds for cross validation.
#' @return A list of numbers, including:
#' \item{pse}{Prediction error of the whole data with cross-validated model.}
#' \item{L.opt}{Optimal number of hidden layers by \code{k}-fold cross-validation.}
#' \item{s.opt}{Optimal factor of L1 regularization by \code{k}-fold cross-validation.}
#' \item{p.opt}{Optimal number of neurons in each layer by \code{k}-fold cross-validation.}
#' @examples
#' ## Two dimension functional data
#' library(pracma)
#' library(MASS)
#' d=2; N=c(3, 5); n=50 
#' ## Generate data
#' Grid=list()
#' Grid[[1]]=seq(1/N[1],1.00,length.out = N[1])
#' Grid[[2]]=seq(1/N[2],1.00,length.out = N[2])
#' x_data.1=as.vector(replicate(N[2], Grid[[1]]))
#' x_data.2=as.vector(t(replicate(N[1], Grid[[2]])))
#' x_train=cbind(x_data.1, x_data.2)
#' ## True function
#' y_train.true=(-8)*1/(1+exp(cot(x_data.1^2)*cos(2*pi*x_data.2)))
#' ## Covariance structure
#' cov=array(NA, c(N[1]*N[2], N[1]*N[2], 2))
#' for(i in 1:(N[1]*N[2])){
#'  for(j in 1:(N[1]*N[2])){
#'    cov[i,j,1]=cos(2*pi*(x_data.1[i] - x_data.1[j]))
#'    cov[i,j,2]=cos(2*pi*(x_data.2[i] - x_data.2[j]))
#'  }
#' }
#' Data=array(NA, c(n, N[1], N[2]))
#' for(i in 1:n){
#'  error=mvrnorm(1, rep(0, N[1]*N[2]), (cov[,,1]+cov[,,2]))+rnorm(1, 0, 1)
#'  Data[i,,]=matrix(error, N[1], N[2])+matrix(y_train.true, N[1], N[2])
#' }
#' FDADNN(Data, d, Grid, N, n, 3, 100, 0.01, 100, 32, 5)
#' @export
FDADNN=function(Data, d, Grid, N, n, L, p, s, epoch, batch, k){
  #Process grid points
  if(d==1){
    x_train=Grid[[1]]
  }else if(d==2){
    x1=rep(rep(Grid[[1]],N[2]))
    x2=rep(Grid[[2]],each=N[1])
    x_train=cbind(x1,x2)
  }else if(d==3){
    x1=rep(Grid[[1]],N[2]*N[3])
    x2=rep(rep(Grid[[2]],each=N[1]),N[3])
    x3=rep(Grid[[3]],each=N[1]*N[2])
    x_train=cbind(x1,x2,x3)
  }else if(d==4){
    x1=rep(rep(Grid[[1]],N[2]*N[3]*N[4]))
    x2=rep(rep(Grid[[2]],each=N[1]),N[3]*N[4])
    x3=rep(rep(Grid[[3]],each=N[1]*N[2]), N[4])
    x4=rep(Grid[[4]],each=N[1]*N[2]*N[3])
    x_train=cbind(x1,x2,x3, x4)
  }
  y_train.raw=matrix(NA, n, base::prod(N))
  for(i in 1:n){
    y_train.raw[i, ]=as.vector(Data[[i]])
  }
  y_train=base::colMeans(y_train.raw)
  #y_train.cv is the average for each fold
  n.cv=n/k
  PSE=array(NA, c(length(L), length(s), length(p)))
  for(ll in 1:length(L)){
    for(ss in 1:length(s)){
      for(pp in 1:length(p)){
        model <- keras::keras_model_sequential() 
        model %>% 
          keras::layer_dense(units = p[pp],  kernel_regularizer = keras::regularizer_l1(s[ss]),  bias_regularizer = keras::regularizer_l1(s[ss]), activation = "relu", input_shape = c(d),kernel_initializer = "normal", keras::constraint_maxnorm(max_value = 1, axis = 0))
        for(xx in  1:L[ll]){
          model %>% keras::layer_dense(units = p[pp],  kernel_regularizer = keras::regularizer_l1(s[ss]),  bias_regularizer = keras::regularizer_l1(s[ss]), activation = "relu",kernel_initializer = "normal", keras::constraint_maxnorm(max_value = 1, axis = 0))
        }
        model %>% keras::layer_dense(units = 1)
        
        model %>% keras::compile(
          loss = "mse",
          optimizer = keras::optimizer_adam(),
          metrics = list("mean_squared_error")
        )
        pse.cv=rep(NA, k)
        for(cv in 1:k){
          index.cv=((cv-1)*n.cv+1): (cv*n.cv)
          y_train.cv=base::colMeans(y_train.raw[-index.cv,], na.rm = FALSE, dims = 1)
          y_test.cv=base::colMeans(y_train.raw[index.cv,], na.rm = FALSE, dims = 1)
          history <- model %>% keras::fit(
            x_train, y_train.cv, 
            epochs = epoch, batch_size = batch
          )
          y.reg.cv=model %>% stats::predict(x_train)
          pse.cv[cv]= mean((y.reg.cv-y_test.cv)^2)
        }
        PSE[ll, ss, pp]=mean(pse.cv)
      }
    }  
  }
  Min=which(PSE==min(PSE), arr.ind = T)
  model <- keras::keras_model_sequential() 
  model %>%
    keras::layer_dense(units = p[Min[3]],  kernel_regularizer = keras::regularizer_l1(s[Min[2]]), bias_regularizer = keras::regularizer_l1(s[Min[2]]), activation = "relu", input_shape = c(d),kernel_initializer = "normal", keras::constraint_maxnorm(max_value = 1, axis = 0)) 
  for(xx in  1:L[Min[1]]){
    model %>% keras::layer_dense(units = p[Min[3]],  kernel_regularizer = keras::regularizer_l1(s[Min[2]]), bias_regularizer = keras::regularizer_l1(s[Min[2]]), activation = "relu",kernel_initializer = "normal", keras::constraint_maxnorm(max_value = 1, axis = 0))
  }
  model %>% keras::layer_dense(units = 1)
  model %>% keras::compile(
    loss = "mse",
    optimizer = keras::optimizer_adam(),
    metrics = list("mean_squared_error")
  )
  history <- model %>% keras::fit(
    x_train, y_train, 
    epochs = epoch, batch_size = batch
  )
  y.reg=model %>% stats::predict(x_train)
  pse=mean((y.reg-y_train)^2)
  L.opt=L[Min[1]]; s.opt=s[Min[2]]; p.opt=p[Min[3]]
  list(pse=pse, L.opt=L.opt, s.opt=s.opt, p.opt=p.opt)
}