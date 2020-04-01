library(ggplot2)

df<-read.csv('output_measures_textsim_norm.csv')
df2<-read.csv("output_multilingual_unisent.csv",sep="\t")
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

dfDataSub<-subset(dfData,metric%in%c("set","cosine_gn300model","word_mover_gn300model","multi-unisent-angdist","multi-unisent-cosine","web-bert"))

dfDataSub$metric<-factor(dfDataSub$metric,levels=c("set","cosine_gn300model","word_mover_gn300model","multi-unisent-cosine","multi-unisent-angdist","web-bert"))

p<-ggplot(dfDataSub,aes(x=measure,fill=which_comparisons))+geom_density(alpha=0.5)
p<-p+facet_wrap(~metric)+theme_bw()+theme(legend.position="bottom",legend.title=element_blank())
p

ggsave("density_plots_subset.png",p,width=12,height=8)
