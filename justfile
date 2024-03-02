test:
    julia --project=. -e 'using Pkg; Pkg.test()'
rerun:
    find . -name "*.jl" | entr julia rerun.jl
resume:
    tmux new-session -d 'just rerun'
    tmux split-window -h 'cd src && nvim -S Session.vim'
    # swap pane locations
    tmux select-pane -t 0
    tmux swap-pane -s 1
    tmux -2 attach-session -d
