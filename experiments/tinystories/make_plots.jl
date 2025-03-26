ENV["GKSwstype"]="nul"
using Glob
using XPlot
using Plots
using Jevo



default(size=(400, 300), margin=5Plots.px)
mediadir = "./media"
# make mediadir if it doesnt exist
isdir(mediadir) || mkdir(mediadir)
struct InteractionDist <: AbstractMetric end

function extract_mean(aggseries)
    TimeSeriesData(name=aggseries.name,data=[XPlot.TimeSeriesDataPoint(p.x, p.mean) for p in aggseries.data], label=aggseries.label)
end


function makeplot(files, met, savename=string(met); ylabel::String=string(met), title=string(met), xlim=:auto, ylim=:auto, hide_dist=false)
    nc = NameConfig(relative_datapath="statistics.h5", seed_suffix="/")
    # for plotting a separate plot for each file
    # xs = [agg(load(met, nc, pi)) for pi in files]
    # for plotting a single plot for all files
    xs = [agg(load(met, nc, files))]
    if hide_dist
        xs = [extract_mean.(xs[1])]
    end

    plots = [plot(x) for x in xs]
    plot(plots..., xlabel="Generation", ylabel=ylabel,title=title, xlim=xlim, ylim=ylim)
    figname=joinpath(mediadir, savename)
    savefig(figname * ".png")
    savefig(figname * ".pdf")
    plot()
end


#= files = [1,2,3,4,5,6] =#
#= paths = String[] =#
#= for i in files =#
#=     append!(paths, glob("0$i*")) =#
#= end =#
#= makeplot(paths, NegativeLoss(), "all_x_means", ylabel="Reward", title="Reward", hide_dist=true) =#

files = [1,2,3]#,4,5,6]
paths = String[]
for i in files
    append!(paths, glob("0$i*"))
end
makeplot(paths, NegativeLoss(), "tfr-loss", ylabel="Negative Cross-Entropy Loss", title="TinyStories Dataset", hide_dist=false, ylim=(-16,-6))
