import matplotlib.pyplot as plt

mylabels={0:'Higgs',1:'QCD (bb)', 2:'QCD (other)'}
histcommon={'histtype':'step','density':True}

def labels(df, varname, labelcol, predcol=None, fmt=None):
    histargs=fmt.hist(varname) if fmt is not None else {}
    histargs['density']=True

    for labelidx in sorted(mylabels.keys()):
        # Plot correctly labeled thing
        sdf=df[df[labelcol]==labelidx]
        _,_,patch=plt.hist(sdf[varname],
                    label=mylabels[labelidx] if predcol is None else None,
                    linestyle='--',
                    **histargs)

        # Plot the predicted thing
        if predcol is not None:
            sdf=df[df[predcol]==labelidx]
            plt.hist(sdf[varname],
                        label=mylabels[labelidx],
                        color=patch[0].get_edgecolor(),
                        **histargs)

    plt.legend()

    fmt.subplot(varname, ax=plt.gca())
    plt.ylabel('normalized')