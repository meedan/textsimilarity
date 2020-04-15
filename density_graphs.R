library(ggplot2)
library(plyr)

df<-read.csv('output_measures_textsim_norm.csv',stringsAsFactors = FALSE)
df2<-read.csv("output_multilingual_unisent.csv",sep="\t",stringsAsFactors = FALSE)
#df3<-read.csv("output_es_match.csv",sep="\t",stringsAsFactors = FALSE)
dfData<-rbind(df,df2)
head(dfData)

p<-ggplot(dfData,aes(x=metric,y=measure,fill=which_comparisons))+geom_boxplot()+coord_flip()
p<-p+theme_bw()+theme(legend.position="bottom",legend.title=element_blank())
ggsave("boxplots.png",p,width=10,height=8)


p<-ggplot(dfData,aes(x=measure,fill=which_comparisons))+geom_density(alpha=0.5)
p<-p+facet_wrap(~metric)+theme_bw()+theme(legend.position="bottom",legend.title=element_blank())
p

ggsave("density_plots.png",p,width=12,height=8)


unique(dfData$metric)

dfDataSub<-subset(dfData,metric%in%c("set","gn300model-cos","gn300model-word-mover","multi-unisent-angdist","multi-unisent-cosine","web-bert"))

dfDataSub$metric<-factor(dfDataSub$metric,levels=c("set","gn300model-cos","gn300model-word-mover","multi-unisent-cosine","multi-unisent-angdist","web-bert"))

p<-ggplot(dfDataSub,aes(x=measure,fill=which_comparisons))+geom_density(alpha=0.5)
p<-p+facet_wrap(~metric)+theme_bw()+theme(legend.position="bottom",legend.title=element_blank())
p

ggsave("density_plots_subset.png",p,width=12,height=8)


##############
# Compare some thresholds

thresholds<-c(1,.99,.97,.95,.9,.85,.8)
metrics<-c("gn300model-cos","gn300model-ang","gn300model-sqrt","multi-unisent-angdist","multi-unisent-cosine","es-match", "set")#,"CR5")

tsub<-subset(dfData,metric%in%metrics)


for (t in thresholds) {
  #dfData[,paste0("t",t)]<-dfData$measure>t
  print("------------------------")
  print(paste0("Threshold: ",t))
  
  vals<-tsub$measure>=t
  sub<-tsub[vals,]
  x<-ddply(sub,.(metric),function(df) {
    return(data.frame(
      len=nrow(df),
      true_pos=sum(df$which_comparisons=="SAME")/nrow(df),
      false_pos=sum(df$which_comparisons=="DIFF")/nrow(df)
    ))
  })
  print(x)
}

print("Total 'good' matches")
sum(tsub$which_comparisons=="SAME" & tsub$metric=="gn300model-cos")
print("Total 'bad' matches")
sum(tsub$which_comparisons=="DIFF" & tsub$metric=="gn300model-cos")
print("Total sentence pairs in data")
sum(tsub$metric=="gn300model-cos")
