using Jevo
using XPlot

FIGNAME = "/tmp/xplot-figs.png"
isfile(FIGNAME) && rm(FIGNAME)
nc = NameConfig(relative_datapath="statistics.h5")
met = XPlot.load(GenotypeSum(), nc, "numbersgame")
plot(met)
XPlot.Plots.savefig(FIGNAME)
!isnothing(Sys.which("kitty")) &&
    run(`kitty +kitten icat $FIGNAME`)
