import glob
import settings
import pandas as pd

def load_data():
    # Load the data
    path=glob.glob(f'{settings.datadir}/user.zhicaiz.309450.NNLOPS_nnlo_30_ggH125_bb_kt200.hbbTrain.e6281_s3126_r10201_p4258.2020_ftag5dev.v0_output.h5/*.output.h5')[0]
    df=pd.read_hdf(path,key='fat_jet')

    # Apply preselection
    df=df[df.nConstituents>2]
    df=df[df.pt>500e3]
    df=df.copy()
    df['mass']=df['mass']/1e3
    df['pt'  ]=df['pt'  ]/1e3

    return df

def label(df):
    """ Add `label` inplace column to dataframe `df`. """
    label0=(df.GhostHBosonsCount==1)
    label1=(df.GhostHBosonsCount==0)&(df.GhostBHadronsFinalCount==2)
    label2=(df.GhostHBosonsCount==0)&(df.GhostBHadronsFinalCount!=2)

    df['label']=3 # default value
    df.loc[label0,'label']=0
    df.loc[label1,'label']=1
    df.loc[label2,'label']=2
