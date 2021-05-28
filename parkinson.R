## CÓDIGO TRABAJO DISFONÍA EN PARKINSON

# Cargar librerías necesarias a priori
library(Rcmdr)
library(tidyverse)
library(colorspace)

# Cargar fichero de datos
path="D:/Miren/Master I/MD/Trabajo final/parkinsons.data"
parkinson <- read.table(path, sep=',', header = TRUE)
parkinson$status<-factor(parkinson$status, 
                         labels=c('Sano','Parkinson'))
# Resumen
summary(parkinson)

# Comprobar varianza de variables
require(caret)
nearZeroVar(parkinson[-c(1,18)], saveMetrics= TRUE)

# Importancia de las variables
rocvarimp2<-filterVarImp(x = parkinson[-c(1,18)], 
                         y = as.factor(parkinson$status))
apply(rocvarimp2, 1, mean) %>% sort()

# Gráfico de correlaciones
require("corrplot")
corrplot((abs(cor(parkinson[-c(1,18)]))>0.95)*0.25+
           (abs(cor(parkinson[-c(1,18)]))>0.9)*0.25+
           (abs(cor(parkinson[-c(1,18)]))>0.85)*0.25+
           (abs(cor(parkinson[-c(1,18)]))>0.8)*0.25, method="circle",
         tl.col = "black")
# Crear nuevo dataset con las variables que interesan
# (Status es el elemento número 9)
parkinson.fil <- parkinson[-c(1, 5, 7, 9, 10, 12, 13, 14, 15, 24)]
parkinson.fil<-parkinson.fil%>%relocate(status)

# Calcular componentes principales
parkinson.PC <- princomp(parkinson.fil[-c(1)], cor=TRUE, scores=TRUE)
summary(parkinson.PC)
# Añadir los PC al dataset
parkinson.fil$PC1<-parkinson.PC$scores[,1]
parkinson.fil$PC2<-parkinson.PC$scores[,2]
parkinson.fil$PC3<-parkinson.PC$scores[,3]
parkinson.fil$PC4<-parkinson.PC$scores[,4]

# Graficar CP por status
scatterplotMatrix(~parkinson.PC$scores[,1:4] | status, regLine=FALSE, smooth=FALSE,
                  diagonal=list(method="density"), by.groups=TRUE,
                  data=parkinson.fil, col=c('#EFC00099','#0073C299'))

# Crear muestras
set.seed(725)
n_data<-195
train<-sample(c(1:n_data), round(0.75 *n_data)) # Muestra de entrenamiento
test<- setdiff(c(1:n_data), train) # Muestra de validación
parkinson.fil.train<-parkinson.fil[train,]
parkinson.fil.test<-parkinson.fil[test,]

# Crear modelo bagging
library(adabag)
set.seed(200323)
parkinson.bagging <- bagging(status~., data=parkinson.fil.train[,1:14],
                             mfinal=50)
# Plotear importancia
a<-rev(sort(parkinson.bagging$importance))
importanceplot(parkinson.bagging,cex.names=0.7, horiz=TRUE)

# Plotear error
errorevol.train <-errorevol(parkinson.bagging, 
                            parkinson.fil.train[,1:14] )
errorevol.test <-errorevol(parkinson.bagging, 
                           parkinson.fil.test[,1:14] )
plot(errorevol.train[[1]], type = "l", xlab = "Iterations", 
     ylab = "Error", col = '#0073C299', lwd = 2, ylim=c(0,0.4))
lines(errorevol.test[[1]], cex = 0.5, col = '#CD534C99', lty = 1,
      lwd = 2)
lines(errorevol.test[[1]]+errorevol.train[[1]], cex = 0.5,
      col = '#EFC00099', lty = 2,lwd = 2)
legend("topright", c("train", "test","train+test"), 
       col=c('#0073C299','#CD534C99','#EFC00099'), lty = c(1,1,2),
       lwd = 2)
# SVM
# Calcular modelo svm óptimo mediante CV
require(e1071)
parkinson.svm.PC<-tune.svm(status~PC1+PC2+PC3+PC4, 
                           data=parkinson.fil.train,
                           coef0=c(0, 0.5, 1,1.5,2.5,5 ), degree=1:5, 
                           cost=c(0.1,1,10))
parkinson.svm.PC$best.parameters
parkinson.svm.PC.1<-svm(status~PC1+PC2+PC3+PC4,
                        data=parkinson.fil.train, 
                        coef0=0, cost=10, degree=1, probability = TRUE)

# Plotear modelo
plot(parkinson.svm.PC.1,data=parkinson.fil.train,
     formula=PC1~PC2,symbolPalette=c('#EFC00099','#CD534C99'),
     color.palette=hsv_palette(h = 204/360, from = 0.9, to = 0.4, v =0.9))

plot(parkinson.svm.PC.1,data=parkinson.fil.test,
     formula=PC1~PC2,symbolPalette=c('#EFC00099','#CD534C99'),
     color.palette=hsv_palette(h = 204/360, from = 0.9, to = 0.4, v =0.9))

# Modelo boosting
library(gbm)
cntrl<-rpart.control(maxdepth=1)
parkinson.boost <- boosting(status ~ .,
                            data=parkinson.fil.train[,1:14],
                            mfinal=200, control=cntrl)

# Plotear importancia
a<-rev(sort(parkinson.boost$importance))
importanceplot(parkinson.boost,cex.names=0.7, horiz=TRUE)
# Plotear error
boost.errorevol.train <-errorevol(parkinson.boost, 
                                  parkinson.fil.train[,1:14] )
boost.errorevol.test <-errorevol(parkinson.boost, 
                                 parkinson.fil.test[,1:14] )
plot(boost.errorevol.train[[1]], type = "l", xlab = "Iterations", 
     ylab = "Error", col = '#0073C299', lwd = 2, ylim=c(0,0.4))
lines(boost.errorevol.test[[1]], cex = 0.5, col = '#CD534C99', lty = 1,
      lwd = 2)
lines(boost.errorevol.test[[1]]+boost.errorevol.train[[1]],
      cex = 0.5, col = '#EFC00099', lty = 2,lwd = 2)
legend("topright", c("train", "test","train+test"), 
       col=c('#0073C299','#CD534C99','#EFC00099'), lty = c(1,1,2),
       lwd = 2)
# Tablas de confusión bagging
# Calcular valores y probabilidades para bagging
bagging.pred.train<-predict(parkinson.bagging,
                            parkinson.fil.train[,1:14], 
                            probability = TRUE)
bagging.pred.train.prob<-bagging.pred.train$prob
bagging.pred.train<-bagging.pred.train$class
bagging.pred.test<-predict(parkinson.bagging,
                           parkinson.fil.test[,1:14], 
                           probability = TRUE)
bagging.pred.test.prob<-bagging.pred.test$prob
bagging.pred.test<-bagging.pred.test$class
# Tablas
xtabs(~parkinson.fil.train$status+bagging.pred.train)
xtabs(~parkinson.fil.test$status+ bagging.pred.test)
# Tablas de confusión SVM
# Calcular valores y probabilidades para SVM
svm.pred.train<-predict(parkinson.svm.PC.1, parkinson.fil.train, 
                        probability = TRUE)
svm.pred.train.prob<-attr(svm.pred.train,"probabilities")
svm.pred.test<-predict(parkinson.svm.PC.1, parkinson.fil.test, 
                       probability = TRUE)
svm.pred.test.prob<-attr(svm.pred.test,"probabilities")
xtabs(~parkinson.fil.train$status+svm.pred.train)
xtabs(~parkinson.fil.test$status+ svm.pred.test)
# Tablas de confusión Boosting
# Probabilidades y predicciones para el modelo Boosting
boost.pred.train<-predict(parkinson.boost,parkinson.fil.train)
boost.pred.test<-predict(parkinson.boost,parkinson.fil.test)
boost.pred.train.prob<-boost.pred.train$prob
boost.pred.test.prob<-boost.pred.test$prob
boost.pred.train<-boost.pred.train$class
boost.pred.test<-boost.pred.test$class
# Tablas
xtabs(~parkinson.fil.train$status+boost.pred.train)
xtabs(~parkinson.fil.test$status+ boost.pred.test)
# Curvas ROC para Bagging
library(pROC)
ROC_bagging_train<-roc(as.numeric(parkinson.fil.train$status),
                       bagging.pred.train.prob[,2],
                       ci=TRUE)
plot(ROC_bagging_train, print.auc=TRUE)
ROC_bagging_test<-roc(as.numeric(parkinson.fil.test$status),
                      bagging.pred.test.prob[,2],
                      ci=TRUE)
plot(ROC_bagging_test, print.auc=TRUE)
# Curvas ROC para SVM
ROC_svm_train<-roc(as.numeric(parkinson.fil.train$status),
                   svm.pred.train.prob[,2],
                   ci=TRUE)
plot(ROC_svm_train, print.auc=TRUE)
ROC_svm_test<-roc(as.numeric(parkinson.fil.test$status),
                  svm.pred.test.prob[,2],
                  ci=TRUE)
plot(ROC_svm_test, print.auc=TRUE)
# Curvas ROC para Boosting
library(pROC)
ROC_boost_train<-roc(as.numeric(parkinson.fil.train$status),
                     boost.pred.train.prob[,2],
                     ci=TRUE)
plot(ROC_boost_train, print.auc=TRUE)
ROC_boost_test<-roc(as.numeric(parkinson.fil.test$status),
                    boost.pred.test.prob[,2],
                    ci=TRUE)
plot(ROC_boost_test, print.auc=TRUE)
# Curvas Lift para Bagging
lift_train<-lift(status ~ bagging.pred.train.prob[,1]+
                   svm.pred.train.prob[,1]+
                   boost.pred.train.prob[,1],
                 data=parkinson.fil.train, class="Sano")
lift_test<-lift(status ~ bagging.pred.test.prob[,1]+
                  svm.pred.test.prob[,1]+
                  boost.pred.test.prob[,1],
                data=parkinson.fil.test, class="Sano")
xyplot(lift_train, auto.key=TRUE,
       plot="gain",col=c('#0073C299','#CD534C99','#EFC00099'),
       key=list(space="top",
                lines=list(col=c('#0073C299','#CD534C99','#EFC00099')), 
                text=list(c("Bagging test", "SVM test", "Boosting test"))))
xyplot(lift_test,
       plot="gain",col=c('#0073C299','#CD534C99','#EFC00099'),
       key=list(space="top",
                lines=list(col=c('#0073C299','#CD534C99','#EFC00099')), 
                text=list(c("Bagging train", "SVM train", 
                            "Boosting train"))))



