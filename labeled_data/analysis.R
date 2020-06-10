library(ggplot2)
library(reshape2)

df<-read.csv("labeled_data_english.csv")
names(df)
nrow(df)

df$FuzzyWuzzy.Score<-df$FuzzyWuzzy.Score/100
mean(df$FuzzyWuzzy.Score,df$LASER.Score)
df$FuzzLaserMean<-df$FuzzyWuzzy.Score/2+df$LASER.Score/2
df$FuzzSbertMean<-df$FuzzyWuzzy.Score/2+df$SBERT.Score/2
df$FuzzSbertMin<-pmin(df$FuzzyWuzzy.Score,df$SBERT.Score)
df$FuzzLaserMin<-pmin(df$FuzzyWuzzy.Score,df$LASER.Score)

dfLong<-melt(df,id.vars=c(names(df)[1:5],"Label"))
names(dfLong)

dfLong<-subset(dfLong,Exclude.Reason=="")

#All items
ggplot(dfLong,aes(x=value,color=Label)) + geom_density() + facet_wrap(~variable) + theme_bw()

##Just for items with at least 10 words
ggplot(subset(dfLong,LookUp.Item.Word.Len>10),aes(x=value,color=Label)) + geom_density() + facet_wrap(~variable) + theme_bw()


#########################
library(PRROC)
methods<-as.character(unique(dfLong$variable))

prsummary<-data.frame()
prcurves<-data.frame()
for (i in 1:length(methods)) {
  print(methods[i])
  tmp<-subset(dfLong,variable==methods[i] & LookUp.Item.Word.Len>3)# & Label!="Somewhat similar")
  tmp$binlabel<-tmp$Label=="Very similar"
  #print(table(tmp$binlabel))
  p<-pr.curve(scores.class0=tmp$value,weights.class0=tmp$binlabel, curve=TRUE)
  #print(p)
  #plot(p)
  tmp<-data.frame(p$curve)
  names(tmp)<-c("recall","precision","threshold")
  tmp$method<-methods[i]
  tmp$method_auc<-paste0(methods[i]," (AUC=",round(p$auc.integral,5),")")
  prcurves<-rbind(prcurves,tmp)
  
  prsummary<-rbind(prsummary,data.frame(
    method=methods[i],
    auc.integral=p$auc.integral,
    auc.davis.goadrich=p$auc.davis.goadrich
  ))
  
}
prsummary
ggplot(prcurves,aes(x=recall,y=precision)) + geom_line() + facet_wrap(~method_auc) + theme_bw()


#How does AUC vary with
#* Excluding / including "Somewhat similar" labels
#* Length



prsummary<-data.frame()
for (Inc.Somewhat in c(TRUE,FALSE)) {
  for (Min.Len in c(0,3,5,7,9)) {
    for (i in 1:length(methods)) {
      #print(methods[i])
      tmp<-subset(dfLong,variable==methods[i] & LookUp.Item.Word.Len>Min.Len)
      if (!Inc.Somewhat) {
          tmp<-subset(tmp,Label!="Somewhat similar")
      }
      tmp$binlabel<-tmp$Label=="Very similar"
      p<-pr.curve(scores.class0=tmp$value,weights.class0=tmp$binlabel, curve=FALSE)
    
      prsummary<-rbind(prsummary,data.frame(
        method=methods[i],
        somewhat=Inc.Somewhat,
        min.len=Min.Len,
        auc.integral=p$auc.integral,
        auc.davis.goadrich=p$auc.davis.goadrich
      ))
    }
  }
}
names(prsummary)
p<-ggplot(prsummary,aes(y=auc.integral,x=method,fill=as.factor(min.len)))
p<-p + geom_bar(position="dodge",stat="identity")
p<-p + facet_wrap(~somewhat) + theme_bw()
p


p<-ggplot(subset(prsummary,min.len==9),aes(y=auc.integral,x=method))
p<-p + geom_bar(position="dodge",stat="identity")
p<-p + facet_wrap(~somewhat) + theme_bw()
p

tmp<-subset(prsummary,min.len==0 & somewhat==TRUE)
tmp[order(tmp$auc.integral),]
